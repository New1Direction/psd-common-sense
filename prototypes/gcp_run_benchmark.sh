#!/usr/bin/env bash
# Upload psd_bench_70b.py, install deps, run benchmark, fetch log.
set -euo pipefail

PROJECT="${GCP_PROJECT:-project-9a9fb340-88e2-413c-860}"
ZONE="${GCP_ZONE:-us-central1-a}"
NAME="${GCP_INSTANCE_NAME:-psd-bench}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="${SCRIPT_DIR}/psd_bench_70b.py"

export CLOUDSDK_AUTH_ACCESS_TOKEN="${CLOUDSDK_AUTH_ACCESS_TOKEN:-$(gcloud auth application-default print-access-token)}"

if [[ ! -f "$BENCH" ]]; then
  echo "Missing $BENCH"
  exit 1
fi

echo "=== nvidia-smi on VM ==="
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" \
  --command="nvidia-smi --query-gpu=index,name,memory.total --format=csv"

echo "=== Upload benchmark ==="
gcloud compute scp "$BENCH" "$NAME":~/psd_bench_70b.py \
  --project="$PROJECT" --zone="$ZONE"

echo "=== Run benchmark (first run downloads ~145GB) ==="
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --command="
  set -e
  pip install -q torch transformers accelerate bitsandbytes huggingface_hub
  if [[ -n \"\${HF_TOKEN:-}\" ]]; then
    huggingface-cli login --token \"\$HF_TOKEN\" --add-to-git-credential
  else
    echo 'Warning: HF_TOKEN not set — gated Llama weights may fail'
  fi
  python3 psd_bench_70b.py 2>&1 | tee ~/psd_bench_70b.log
"

gcloud compute scp "$NAME":~/psd_bench_70b.log "$SCRIPT_DIR/psd_bench_70b.log" \
  --project="$PROJECT" --zone="$ZONE"

echo "=== Log saved to $SCRIPT_DIR/psd_bench_70b.log ==="
tail -40 "$SCRIPT_DIR/psd_bench_70b.log"