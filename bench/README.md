# PSD benchmark runner

Turns a multi-step agent trace into a **measured** PSD result: per-step GPU draft
latency, local CSE verdict latency, prefetch hit rate, and verdict mix — written to CSV
plus a copy-pasteable summary table.

It deliberately separates **measured** numbers (latency, hit rate) from **projected**
ones (`savings = hitRate × avgDraftLatency` is a model, not a wall-clock measurement).

## Offline dry run (no GPU)

Validates the whole pipeline with a simulated draft/verify endpoint. Latencies are
**simulated**, so don't quote them — this is for wiring and CI, not for results.

```bash
npm run bench -- --offline
```

## Real run (Vast box)

1. Start two vLLM servers on the GPU box (one per role; they share the card):

   ```bash
   vllm serve Qwen/Qwen2.5-14B-Instruct --port 18000 --max-model-len 8192 --gpu-memory-utilization 0.45 &
   vllm serve Qwen/Qwen2.5-14B-Instruct --port 18001 --max-model-len 8192 --gpu-memory-utilization 0.45 &
   ```

2. Tunnel both ports locally:

   ```bash
   ssh -i ~/.ssh/your-key -o IdentitiesOnly=yes \
     -L 18000:127.0.0.1:18000 -L 18001:127.0.0.1:18001 \
     -p <vast-port> root@<vast-host> -N &
   ```

3. Run the benchmark over a real trace (≥30 steps is where the hit rate becomes a rate,
   not an anecdote — repeat the bundled trace or supply your own):

   ```bash
   npm run bench -- --trace bench/sample-trace.json --out bench/results.csv
   # grow the step count cheaply:
   npm run bench -- --repeat 3 --out bench/results.csv
   ```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--trace <path>` | bundled sample | Trace JSON: array of `{action, goal?, state?}` (or bare action strings). |
| `--out <path>` | `bench/results.csv` | Per-step CSV output. |
| `--goal <text>` | build-fix goal | Overall goal applied when a step omits its own. |
| `--repeat <n>` | `1` | Repeat the trace n times to grow the sample. |
| `--max-branches <n>` | `4` | Branches drafted per step. |
| `--offline` | off | Simulated GPU; no endpoints needed. |
| `--sim-draft-ms` / `--sim-verify-ms` | `40` / `6` | Simulated latencies in offline mode. |
| `--draft-url` / `--verifier-url` | env / localhost | Override vLLM base URLs. |

## Trace format

```json
[
  { "action": "run the test suite", "state": { "branch": "fix/x", "tests_passing": false } },
  { "action": "git push --force origin main", "state": { "branch": "main" } }
]
```

Each step's `state` is the world state the drafted branches are judged against, so vary it
to exercise the full `ready → warn → ask_human → blocked` range.
