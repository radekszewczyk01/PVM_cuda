#!/usr/bin/env bash
# docker_build.sh – Build the PVM CUDA C Docker image
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Build context is the repo root so the Dockerfile can COPY model_zoo
cd "$SCRIPT_DIR/.."

# Auto-detect GPU arch for the build container; fall back to sm_75
GPU=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | tr -d '.' | sed 's/^/sm_/')
[ "$GPU" = "sm_" ] && GPU="sm_75"
echo "Using GPU arch: $GPU"

IMAGE="${IMAGE:-pvm_cuda_c:latest}"
echo "Building Docker image: $IMAGE"
docker build -t "$IMAGE" -f cuda_native_c/Dockerfile --build-arg GPU="$GPU" .
echo "Done: $IMAGE"
