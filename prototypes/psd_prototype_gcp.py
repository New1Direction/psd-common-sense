"""
PSD Prototype — GCP benchmark (Llama 3.1 8B target + Llama 3.2 1B draft)
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field

DEVICE = "cuda"

# ── config ─────────────────────────────────────────────────────────────────
TARGET_MODEL = "meta-llama/Llama-3.1-8B"
DRAFT_MODEL = "meta-llama/Llama-3.2-1B"

K = 4
N_STEPS = 20
TOP_P = 0.9
TEMP = 1.0
RUNNERUP_K = 5

PROMPTS = [
    "The key insight about transformer models is",
    "In order to optimize inference speed we need to",
    "The future of large language models depends on",
    "Speculative decoding works by having a small model",
]


@dataclass
class Stats:
    bonus_hits: int = 0
    bonus_misses: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    steps: int = 0
    psd_tokens: list = field(default_factory=list)
    baseline_tokens: list = field(default_factory=list)

    @property
    def hit_rate(self):
        total = self.bonus_hits + self.bonus_misses
        return self.bonus_hits / total if total > 0 else 0.0

    @property
    def accept_rate(self):
        total = self.tokens_accepted + self.tokens_rejected
        return self.tokens_accepted / total if total > 0 else 0.0


def draft_tokens(model, input_ids, k: int, temperature: float = 1.0):
    draft_ids = []
    draft_logits = []
    cur_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(k):
            out = model(cur_ids)
            logits = out.logits[:, -1, :]
            draft_logits.append(logits.squeeze(0))

            scaled = logits / temperature
            probs = F.softmax(scaled, dim=-1)
            token = torch.multinomial(probs, 1)

            draft_ids.append(token.squeeze())
            cur_ids = torch.cat([cur_ids, token], dim=-1)

    return torch.stack(draft_ids), torch.stack(draft_logits)


def predict_bonus_token(draft_logits_last, runnerup_k: int = 5):
    probs = F.softmax(draft_logits_last, dim=-1)
    top_vals, top_ids = torch.topk(probs, runnerup_k + 1)
    runnerup_probs = top_vals[1:]
    runnerup_probs = runnerup_probs / runnerup_probs.sum()
    idx = torch.multinomial(runnerup_probs, 1).item()
    predicted_bonus = top_ids[idx + 1].item()
    candidate_set = set(top_ids[1:].tolist())
    return predicted_bonus, candidate_set


def verify_tokens(target_model, prefix_ids, draft_ids, temperature: float = 1.0):
    full_ids = torch.cat([prefix_ids, draft_ids.unsqueeze(0)], dim=-1)

    with torch.no_grad():
        out = target_model(full_ids)

    prefix_len = prefix_ids.shape[-1]
    verify_logits = out.logits[0, prefix_len - 1 : prefix_len + len(draft_ids), :]

    accepted = []
    bonus_token = None

    for i, draft_tok in enumerate(draft_ids):
        target_logits = verify_logits[i]
        scaled = target_logits / temperature
        probs = F.softmax(scaled, dim=-1)
        target_tok = torch.multinomial(probs, 1).item()

        if target_tok == draft_tok.item():
            accepted.append(draft_tok.item())
        else:
            bonus_token = target_tok
            break

    if bonus_token is None:
        bonus_logits = verify_logits[len(draft_ids)]
        scaled = bonus_logits / temperature
        probs = F.softmax(scaled, dim=-1)
        bonus_token = torch.multinomial(probs, 1).item()

    return accepted, bonus_token


def run_psd(target_model, draft_model, tokenizer, prompt: str, stats: Stats):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = input_ids.clone()
    psd_token_count = 0

    for _ in range(N_STEPS):
        draft_ids, draft_logits = draft_tokens(draft_model, generated, k=K, temperature=TEMP)
        predicted_bonus, _ = predict_bonus_token(draft_logits[-1], runnerup_k=RUNNERUP_K)
        accepted, actual_bonus = verify_tokens(target_model, generated, draft_ids, temperature=TEMP)

        if predicted_bonus == actual_bonus:
            stats.bonus_hits += 1
        else:
            stats.bonus_misses += 1

        new_tokens = accepted + [actual_bonus]
        new_ids = torch.tensor(new_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
        generated = torch.cat([generated, new_ids], dim=-1)

        stats.tokens_accepted += len(accepted)
        stats.tokens_rejected += K - len(accepted)
        stats.steps += 1
        psd_token_count += len(new_tokens)

    stats.psd_tokens.append(psd_token_count)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def run_baseline(target_model, tokenizer, prompt: str, stats: Stats):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    total_tokens = K * N_STEPS

    with torch.no_grad():
        out = target_model.generate(
            input_ids,
            max_new_tokens=total_tokens,
            do_sample=True,
            temperature=TEMP,
        )

    stats.baseline_tokens.append(total_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=dtype, device_map="cuda"
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, torch_dtype=dtype, device_map="cuda"
    )
    target_model.eval()
    draft_model.eval()

    print(f"  target: {TARGET_MODEL}")
    print(f"  draft:  {DRAFT_MODEL}")
    print(f"  k={K}  steps={N_STEPS}  runnerup_k={RUNNERUP_K}\n")

    stats = Stats()
    psd_times = []
    baseline_times = []

    for i, prompt in enumerate(PROMPTS):
        print(f"Prompt {i+1}: \"{prompt[:50]}...\"")

        t0 = time.perf_counter()
        psd_out = run_psd(target_model, draft_model, tokenizer, prompt, stats)
        psd_ms = (time.perf_counter() - t0) * 1000
        psd_times.append(psd_ms)

        t0 = time.perf_counter()
        _ = run_baseline(target_model, tokenizer, prompt, stats)
        base_ms = (time.perf_counter() - t0) * 1000
        baseline_times.append(base_ms)

        print(f"  PSD:      {psd_ms:7.0f}ms")
        print(f"  Baseline: {base_ms:7.0f}ms")
        print(f"  Ratio:    {base_ms/psd_ms:.2f}x\n")

    print("═" * 60)
    print("RESULTS")
    print("═" * 60)
    print(
        f"Bonus token hit rate:    {stats.hit_rate:.1%}  "
        f"({stats.bonus_hits}/{stats.bonus_hits + stats.bonus_misses} steps)"
    )
    print(
        f"Draft accept rate:       {stats.accept_rate:.1%}  "
        f"({stats.tokens_accepted}/{stats.tokens_accepted + stats.tokens_rejected} tokens)"
    )
    print(f"Avg PSD time/prompt:     {sum(psd_times)/len(psd_times):.0f}ms")
    print(f"Avg baseline time:       {sum(baseline_times)/len(baseline_times):.0f}ms")
    print()
    print("Sample PSD output (last prompt):")
    print(f"  {psd_out[:200]}...")


if __name__ == "__main__":
    main()