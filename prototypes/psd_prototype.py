"""
PSD Prototype — Predictive Speculative Decoding
Validates the core logic on CPU with tiny models.
Measures: bonus token prediction accuracy, tokens/sec vs baseline.

Usage:
    python psd_prototype.py
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import Optional

# ── config ─────────────────────────────────────────────────────────────────
TARGET_MODEL = "facebook/opt-125m"   # acts as "target" (verifier)
DRAFT_MODEL  = "facebook/opt-125m"   # same model — logic test, not speed test
# on GCP swap to:
# TARGET_MODEL = "meta-llama/Llama-3.1-8B"
# DRAFT_MODEL  = "meta-llama/Llama-3.2-1B"

K          = 4      # draft tokens per step
N_STEPS    = 20     # steps per run
TOP_P      = 0.9    # sampling for draft
TEMP       = 1.0    # temperature
RUNNERUP_K = 5      # how many runner-up tokens to consider for bonus prediction

PROMPTS = [
    "The key insight about transformer models is",
    "In order to optimize inference speed we need to",
    "The future of large language models depends on",
    "Speculative decoding works by having a small model",
]

# ── stats tracker ──────────────────────────────────────────────────────────
@dataclass
class Stats:
    bonus_hits:   int = 0
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


# ── core PSD functions ──────────────────────────────────────────────────────

def draft_tokens(model, input_ids, k: int, temperature: float = 1.0):
    """
    Draft model generates k tokens autoregressively.
    Returns:
        draft_ids:    [k] tensor of sampled token ids
        draft_logits: [k, vocab] tensor of logits at each step
    """
    draft_ids = []
    draft_logits = []
    cur_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(k):
            out = model(cur_ids)
            logits = out.logits[:, -1, :]          # [1, vocab]
            draft_logits.append(logits.squeeze(0))  # [vocab]

            scaled = logits / temperature
            probs  = F.softmax(scaled, dim=-1)
            token  = torch.multinomial(probs, 1)    # [1, 1]

            draft_ids.append(token.squeeze())
            cur_ids = torch.cat([cur_ids, token], dim=-1)

    return torch.stack(draft_ids), torch.stack(draft_logits)


def predict_bonus_token(draft_logits_last, runnerup_k: int = 5):
    """
    PSD key mechanism: predict what the target model will output as the
    bonus token by sampling from the draft model's runner-up distribution.

    The insight: when target disagrees with draft at position k,
    it tends to pick from the draft's high-probability alternatives.

    Returns:
        predicted_bonus: token id (int)
        candidate_set:   set of top-k token ids considered
    """
    probs = F.softmax(draft_logits_last, dim=-1)

    # get top-k runner-up tokens (exclude the argmax — that's what we already drafted)
    top_vals, top_ids = torch.topk(probs, runnerup_k + 1)

    # sample from the runner-up distribution (skip idx 0 = the sampled token)
    runnerup_probs = top_vals[1:]
    runnerup_probs = runnerup_probs / runnerup_probs.sum()  # renormalize
    idx = torch.multinomial(runnerup_probs, 1).item()

    predicted_bonus = top_ids[idx + 1].item()
    candidate_set   = set(top_ids[1:].tolist())

    return predicted_bonus, candidate_set


def verify_tokens(target_model, prefix_ids, draft_ids, temperature: float = 1.0):
    """
    Target model verifies all k draft tokens in ONE forward pass.
    Returns:
        accepted:    list of accepted token ids
        bonus_token: the target's own next token after last accept
        n_accepted:  how many draft tokens were accepted
    """
    full_ids = torch.cat([prefix_ids, draft_ids.unsqueeze(0)], dim=-1)

    with torch.no_grad():
        out = target_model(full_ids)

    # logits at positions len(prefix)..len(prefix)+k  (one per draft token + bonus)
    prefix_len   = prefix_ids.shape[-1]
    verify_logits = out.logits[0, prefix_len - 1 : prefix_len + len(draft_ids), :]

    accepted   = []
    bonus_token = None

    for i, draft_tok in enumerate(draft_ids):
        target_logits = verify_logits[i]
        scaled = target_logits / temperature
        probs  = F.softmax(scaled, dim=-1)
        target_tok = torch.multinomial(probs, 1).item()

        if target_tok == draft_tok.item():
            accepted.append(draft_tok.item())
        else:
            # first mismatch — target inserts its own token (the "bonus")
            bonus_token = target_tok
            break

    if bonus_token is None:
        # all k accepted — target generates one more bonus token
        bonus_logits = verify_logits[len(draft_ids)]
        scaled = bonus_logits / temperature
        probs  = F.softmax(scaled, dim=-1)
        bonus_token = torch.multinomial(probs, 1).item()

    return accepted, bonus_token


def run_psd(target_model, draft_model, tokenizer, prompt: str, stats: Stats):
    """Full PSD loop for one prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids.clone()
    psd_token_count = 0

    for step in range(N_STEPS):
        # 1. draft model generates k tokens + records logits
        draft_ids, draft_logits = draft_tokens(
            draft_model, generated, k=K, temperature=TEMP
        )

        # 2. PSD: predict bonus token from runner-up distribution
        #    (in real async impl this fires WHILE target is verifying)
        predicted_bonus, candidate_set = predict_bonus_token(
            draft_logits[-1], runnerup_k=RUNNERUP_K
        )

        # 3. target verifies all k draft tokens in one pass
        accepted, actual_bonus = verify_tokens(
            target_model, generated, draft_ids, temperature=TEMP
        )

        # 4. track bonus prediction accuracy
        if predicted_bonus == actual_bonus:
            stats.bonus_hits += 1
        else:
            stats.bonus_misses += 1

        # 5. commit: append accepted tokens + bonus
        new_tokens = accepted + [actual_bonus]
        new_ids    = torch.tensor(new_tokens, dtype=torch.long).unsqueeze(0)
        generated  = torch.cat([generated, new_ids], dim=-1)

        stats.tokens_accepted += len(accepted)
        stats.tokens_rejected += (K - len(accepted))
        stats.steps += 1
        psd_token_count += len(new_tokens)

    stats.psd_tokens.append(psd_token_count)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def run_baseline(target_model, tokenizer, prompt: str, stats: Stats):
    """Vanilla autoregressive baseline — same number of tokens."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    total_tokens = K * N_STEPS  # match PSD output length roughly

    with torch.no_grad():
        out = target_model.generate(
            input_ids,
            max_new_tokens=total_tokens,
            do_sample=True,
            temperature=TEMP,
        )

    stats.baseline_tokens.append(total_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading models...")
    tokenizer    = AutoTokenizer.from_pretrained(TARGET_MODEL)
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, torch_dtype=torch.float32)
    draft_model  = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL,  torch_dtype=torch.float32)
    target_model.eval()
    draft_model.eval()
    print(f"  target: {TARGET_MODEL}")
    print(f"  draft:  {DRAFT_MODEL}")
    print(f"  k={K}  steps={N_STEPS}  runnerup_k={RUNNERUP_K}\n")

    stats = Stats()

    print("─" * 60)
    psd_times      = []
    baseline_times = []

    for i, prompt in enumerate(PROMPTS):
        print(f"Prompt {i+1}: \"{prompt[:50]}...\"")

        # PSD run
        t0  = time.perf_counter()
        psd_out = run_psd(target_model, draft_model, tokenizer, prompt, stats)
        psd_ms  = (time.perf_counter() - t0) * 1000
        psd_times.append(psd_ms)

        # baseline run
        t0  = time.perf_counter()
        _   = run_baseline(target_model, tokenizer, prompt, stats)
        base_ms = (time.perf_counter() - t0) * 1000
        baseline_times.append(base_ms)

        print(f"  PSD:      {psd_ms:7.0f}ms")
        print(f"  Baseline: {base_ms:7.0f}ms")
        print(f"  Ratio:    {base_ms/psd_ms:.2f}x  (on same model — logic test only)\n")

    # ── results ──────────────────────────────────────────────────────────
    print("═" * 60)
    print("RESULTS")
    print("═" * 60)
    print(f"Bonus token hit rate:    {stats.hit_rate:.1%}  ({stats.bonus_hits}/{stats.bonus_hits+stats.bonus_misses} steps)")
    print(f"Draft accept rate:       {stats.accept_rate:.1%}  ({stats.tokens_accepted}/{stats.tokens_accepted+stats.tokens_rejected} tokens)")
    print(f"Avg PSD time/prompt:     {sum(psd_times)/len(psd_times):.0f}ms")
    print(f"Avg baseline time:       {sum(baseline_times)/len(baseline_times):.0f}ms")
    print()
    print("NOTE: timing on same model = meaningless for speed.")
    print("      Bonus hit rate is the number that matters here.")
    print()

    if stats.hit_rate >= 0.75:
        print("✓ Hit rate looks good — PSD logic is sound.")
        print("  Next step: GCP with real draft+target pair for real speedup numbers.")
    else:
        print("△ Hit rate lower than expected.")
        print("  Try: increase RUNNERUP_K, or use a draft model from the same family.")

    print()
    print("Sample PSD output (last prompt):")
    print(f"  {psd_out[:200]}...")


if __name__ == "__main__":
    main()
