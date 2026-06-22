#!/usr/bin/env bash
# Create psd-bench on a2-highgpu-2g (2x A100 40GB). Needs GPU quota > 0.
set -euo pipefail

PROJECT="${GCP_PROJECT:-project-9a9fb340-88e2-413c-860}"
ZONE="${GCP_ZONE:-us-central1-a}"
NAME="${GCP_INSTANCE_NAME:-psd-bench}"
MACHINE="${GCP_MACHINE_TYPE:-a2-highgpu-2g}"

export CLOUDSDK_AUTH_ACCESS_TOKEN="${CLOUDSDK_AUTH_ACCESS_TOKEN:-$(gcloud auth application-default print-access-token)}"

IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"

echo "Creating ${NAME} (${MACHINE}) in ${ZONE}..."

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --metadata=install-nvidia-driver=True \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo "Done. SSH: gcloud compute ssh $NAME --project=$PROJECT --zone=$ZONE"