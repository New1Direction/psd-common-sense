# PSD Common Sense

**Predictive Safety Decoding (PSD)** + **Common Sense Engine (CSE)** for agent workflows.

PSD hides next-step branch drafting behind current-step CSE judgment — the same latency-hiding principle as [SSD (Speculative Speculative Decoding)](https://arxiv.org/abs/2603.03251), applied at the **agent action** layer instead of token inference.

## Benchmark

Tested on 2× A100 SXM4 (Vast.ai), Qwen2.5-14B-Instruct, coding agent task.

| Metric | Value |
|---|---|
| Prefetch hit rate | 66.7% (2/3 steps) |
| Avg draft latency | 3881ms |
| Projected savings/step | ~2589ms hidden per step |
| Blocked branches | 0 |
| CSE verdict latency | <2ms (local, deterministic) |

### Comparison to SSD (token decoding)

| | SSD (Tanishq et al., 2026) | PSD — this work |
|---|---|---|
| Layer | Token inference | Agent action planning |
| Verifier | Target LLM (70B) | Common Sense Engine (local) |
| Draft model | Small LLM (1-3B) | GPU branch generator (14B) |
| Parallelism | Draft overlaps verify | Branch gen overlaps CSE |
| Hit rate | ~85% token acceptance | 66.7% branch prefetch |
| Latency hidden | Verification RTT | Full draft RTT (~3.8s) |

Same latency-hiding principle, different layer.
SSD hides token verification behind draft speculation.
PSD hides next-step branch drafting behind current-step CSE judgment.

## Architecture

| Layer | Role |
|-------|------|
| **GPU imagination** (`vast-client`) | Draft + verifier vLLM on ports 18000 / 18001 |
| **CSE** (`common-sense`, `cse-verdict`) | Local deterministic verdicts: `ready`, `warn`, `ask_human`, `blocked` |
| **PSD scheduler** (`psd-scheduler`) | Parallel branch verify + prefetch with benchmark metrics |

## Quick start

```bash
npm install
npm test

# SSH tunnel to Vast vLLM (example)
ssh -i ~/.ssh/vast_codex_ed25519 -o IdentitiesOnly=yes \
  -L 18000:127.0.0.1:18000 -L 18001:127.0.0.1:18001 \
  -p 58764 root@<host> -N &

npm run demo
```

```ts
import { runPsdScheduler, createVastImaginationClient } from "psd-common-sense";

const result = await runPsdScheduler({
  current: {
    action: "run tests",
    goal: "fix broken TypeScript file with failing tests",
    state: { tests_passing: false }
  },
  imagination: {},
  parallelPrefetch: true
});

console.log(result.metrics);
```

## Modules

- `src/common-sense.ts` — CSE forecasts and rule packs
- `src/cse-verdict.ts` — `approve` ↔ `ready` mapping and `scoreCseVerdict()`
- `src/vast-client.ts` — GPU branch drafter + verifier client
- `src/vast-mcp.ts` — Vast.ai instance / health MCP helpers
- `src/psd-scheduler.ts` — PSD loop and metrics
- `src/psd-agent.ts` — Demo entry (`npm run demo`)
- `src/predictive-safety-decoding.ts` — Branch ranking and commit strategy

## License

MIT