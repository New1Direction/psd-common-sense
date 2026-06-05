# PSD Python sidecar (optional)

The canonical PSD loop lives in TypeScript:

- `src/vast-client.ts` — imagination layer (draft + verifier vLLM)
- `src/cse-verdict.ts` — local deterministic judgment (`ready` / `warn` / `ask_human` / `blocked`)
- `src/psd-scheduler.ts` — parallel verify + prefetch with metrics

If you still have the external Python modules (`vast_client.py`, `cse.py`, `scheduler.py`, `agent.py`), place them in this directory and run:

```bash
export VAST_DRAFT_BASE_URL=http://127.0.0.1:18000/v1
export VAST_VERIFIER_BASE_URL=http://127.0.0.1:18001/v1
python agent.py
```

SSH tunnel (from handoff):

```bash
ssh -i ~/.ssh/vast_codex_ed25519 -o IdentitiesOnly=yes \
  -L 18000:127.0.0.1:18000 -L 18001:127.0.0.1:18001 \
  -p 58764 root@159.48.242.6 -N
```

TypeScript clients use the same URLs via `createVastImaginationClient()` or `VAST_DRAFT_BASE_URL` / `VAST_VERIFIER_BASE_URL`.