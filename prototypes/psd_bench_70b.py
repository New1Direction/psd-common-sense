"""
PSD 70B benchmark — 2x A100 (a2-highgpu-2g)
  GPU 0: Llama 3.3 70B (first shard, ~38GB, 4-bit)
  GPU 1: Llama 3.3 70B (second shard) + Llama 3.2 3B draft (~6GB)

Async: target verify runs across both GPUs while bonus prediction
runs on draft logits (GPU 1). Uses CUDA streams to overlap work.

Usage:
  python psd_bench_70b.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── config ─────────────────────────────────────────────────────────────────
TARGET_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DRAFT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DRAFT_DEVICE = 1

K = 4
N_STEPS = 20
TEMP = 1.0
RUNNERUP_K = 8

PROMPTS = [
    "The key insight about transformer models is",
    "In order to optimize inference speed we need to",
    "The future of large language models depends on",
    "Speculative decoding works by having a small model",
    "Parallel speculative decoding hides latency by predicting",
    "When the draft model disagrees with the target verifier,",
]

# Reserve headroom on GPU 1 for the 3B draft (~6GB) before loading 70B shards
MAX_MEMORY = {0: "38GiB", 1: "32GiB"}


@dataclass
class Stats:
    bonus_hits: int = 0
    bonus_misses: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    steps: int = 0

    @property
    def hit_rate(self) -> float:
        t = self.bonus_hits + self.bonus_misses
        return self.bonus_hits / t if t else 0.0

    @property
    def accept_rate(self) -> float:
        t = self.tokens_accepted + self.tokens_rejected
        return self.tokens_accepted / t if t else 0.0


def print_gpu_memory(label: str = "") -> None:
    print(f"  [{label}] VRAM:")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"    GPU {i}: {alloc:.1f}GB allocated, {reserved:.1f}GB reserved")


def draft_tokens(draft_model, input_ids: torch.Tensor, k: int, stream: torch.cuda.Stream):
    """Draft k tokens on GPU 1 (3B)."""
    draft_ids = []
    draft_logits = []
    cur = input_ids.to(DRAFT_DEVICE)

    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(k):
            out = draft_model(cur)
            logits = out.logits[:, -1, :].squeeze(0)
            draft_logits.append(logits)
            probs = F.softmax(logits / TEMP, dim=-1)
            token = torch.multinomial(probs, 1).view(1, 1)
            draft_ids.append(token.squeeze())
            cur = torch.cat([cur, token], dim=-1)

    stream.synchronize()
    return torch.stack(draft_ids), torch.stack(draft_logits)


def predict_bonus_token(draft_logits_last: torch.Tensor, runnerup_k: int = RUNNERUP_K):
    probs = F.softmax(draft_logits_last.float(), dim=-1)
    top_vals, top_ids = torch.topk(probs, runnerup_k + 1)
    runnerup_probs = top_vals[1:]
    runnerup_probs = runnerup_probs / runnerup_probs.sum()
    idx = torch.multinomial(runnerup_probs, 1).item()
    return top_ids[idx + 1].item()


def verify_tokens(target_model, prefix_ids: torch.Tensor, draft_ids: torch.Tensor):
    """Single target forward over prefix + k drafts; runs on sharded 70B."""
    device = prefix_ids.device
    full = torch.cat([prefix_ids, draft_ids.unsqueeze(0).to(device)], dim=-1)

    with torch.no_grad():
        out = target_model(full)

    prefix_len = prefix_ids.shape[-1]
    vlog = out.logits[0, prefix_len - 1 : prefix_len + len(draft_ids), :]

    accepted = []
    bonus_token = None

    for i, draft_tok in enumerate(draft_ids):
        logits = vlog[i]
        probs = F.softmax(logits / TEMP, dim=-1)
        target_tok = torch.multinomial(probs, 1).item()
        if target_tok == int(draft_tok.item()):
            accepted.append(target_tok)
        else:
            bonus_token = target_tok
            break

    if bonus_token is None:
        logits = vlog[len(draft_ids)]
        probs = F.softmax(logits / TEMP, dim=-1)
        bonus_token = torch.multinomial(probs, 1).item()

    return accepted, bonus_token


def run_psd_async(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    stats: Stats,
    draft_stream: torch.cuda.Stream,
) -> str:
    """
    PSD loop with async overlap:
      - verify (70B, both GPUs) runs in a worker thread
      - predict_bonus (draft logits) runs in parallel on GPU 1
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = next(target_model.parameters()).device
    generated = input_ids.to(device)

    for _ in range(N_STEPS):
        draft_ids, draft_logits = draft_tokens(
            draft_model, generated, K, draft_stream
        )

        predicted_holder: list = []
        verify_holder: list = []
        err_holder: list = []

        def _predict():
            try:
                with torch.cuda.device(DRAFT_DEVICE):
                    predicted_holder.append(
                        predict_bonus_token(draft_logits[-1].detach())
                    )
            except Exception as e:
                err_holder.append(e)

        def _verify():
            try:
                verify_holder.append(
                    verify_tokens(target_model, generated, draft_ids.cpu())
                )
            except Exception as e:
                err_holder.append(e)

        # Overlap target verify (both GPUs) with bonus prediction (GPU 1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_pred = pool.submit(_predict)
            f_verify = pool.submit(_verify)
            f_pred.result()
            f_verify.result()

        if err_holder:
            raise err_holder[0]

        predicted_bonus = predicted_holder[0]
        accepted, actual_bonus = verify_holder[0]

        if predicted_bonus == actual_bonus:
            stats.bonus_hits += 1
        else:
            stats.bonus_misses += 1

        new_tokens = accepted + [actual_bonus]
        new_ids = torch.tensor(new_tokens, dtype=torch.long, device=device).unsqueeze(0)
        generated = torch.cat([generated, new_ids], dim=-1)

        stats.tokens_accepted += len(accepted)
        stats.tokens_rejected += K - len(accepted)
        stats.steps += 1

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def run_baseline(target_model, tokenizer, prompt: str) -> None:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = next(target_model.parameters()).device
    input_ids = input_ids.to(device)
    total = K * N_STEPS

    with torch.no_grad():
        target_model.generate(
            input_ids,
            max_new_tokens=total,
            do_sample=True,
            temperature=TEMP,
        )


def load_models():
    print("Loading draft model (3B on GPU 1)...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL,
        torch_dtype=torch.float16,
        device_map={"": DRAFT_DEVICE},
    )
    draft_model.eval()
    print_gpu_memory("after draft")

    print("Loading target model (70B 4-bit, sharded auto)...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        quantization_config=quant,
        device_map="auto",
        max_memory=MAX_MEMORY,
    )
    target_model.eval()
    print_gpu_memory("after target")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return target_model, draft_model, tokenizer


def main():
    assert torch.cuda.device_count() >= 2, "Need 2 GPUs (a2-highgpu-2g)"
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    target_model, draft_model, tokenizer = load_models()
    draft_stream = torch.cuda.Stream(device=DRAFT_DEVICE)

    print(f"\nTarget: {TARGET_MODEL}")
    print(f"Draft:  {DRAFT_MODEL}")
    print(f"k={K} steps={N_STEPS} runnerup_k={RUNNERUP_K}\n")
    print("─" * 60)

    stats = Stats()
    psd_times: list[float] = []
    base_times: list[float] = []
    speedups: list[float] = []

    for i, prompt in enumerate(PROMPTS):
        print(f"Prompt {i + 1}/{len(PROMPTS)}: \"{prompt[:55]}...\"")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_psd_async(
            target_model, draft_model, tokenizer, prompt, stats, draft_stream
        )
        torch.cuda.synchronize()
        psd_ms = (time.perf_counter() - t0) * 1000
        psd_times.append(psd_ms)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_baseline(target_model, tokenizer, prompt)
        torch.cuda.synchronize()
        base_ms = (time.perf_counter() - t0) * 1000
        base_times.append(base_ms)

        speedup = base_ms / psd_ms if psd_ms > 0 else 0.0
        speedups.append(speedup)

        print(f"  PSD:      {psd_ms:8.0f} ms")
        print(f"  Baseline: {base_ms:8.0f} ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print(
            f"  Running hit rate: {stats.hit_rate:.1%}  "
            f"accept: {stats.accept_rate:.1%}\n"
        )

    print("═" * 60)
    print("RESULTS (70B + 3B, 2x A100)")
    print("═" * 60)
    print(
        f"Bonus token hit rate:    {stats.hit_rate:.1%}  "
        f"({stats.bonus_hits}/{stats.bonus_hits + stats.bonus_misses} steps)"
    )
    print(
        f"Draft accept rate:       {stats.accept_rate:.1%}  "
        f"({stats.tokens_accepted}/{stats.tokens_accepted + stats.tokens_rejected} tokens)"
    )
    print(f"Avg PSD time/prompt:     {sum(psd_times) / len(psd_times):.0f} ms")
    print(f"Avg baseline time:       {sum(base_times) / len(base_times):.0f} ms")
    print(f"Avg speedup:             {sum(speedups) / len(speedups):.2f}x")
    print(f"Per-prompt speedups:     {[round(s, 2) for s in speedups]}")

    if stats.hit_rate >= 0.75 and sum(speedups) / len(speedups) >= 1.5:
        print("\n✓ Tweet-ready: hit rate 75%+ and speedup 1.5x+")
    elif stats.hit_rate >= 0.75:
        print("\n△ Hit rate good; speedup below 1.5x — tune k/steps or async overlap")
    else:
        print("\n△ Try increasing RUNNERUP_K (10–15) if hit rate is low")

    print_gpu_memory("final")


if __name__ == "__main__":
    main()