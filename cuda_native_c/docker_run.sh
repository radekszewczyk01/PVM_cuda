#!/usr/bin/env bash
# docker_run.sh – Run PVM CUDA C inside Docker with GPU and volume mounts
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE="${IMAGE:-pvm_cuda_c:latest}"
SAVES_DIR="${SAVES_DIR:-$SCRIPT_DIR/saves}"
DATA_DIR="${DATA_DIR:-}"

mkdir -p "$SAVES_DIR"

extra_vols=""
[ -n "$DATA_DIR" ] && extra_vols="-v $DATA_DIR:/data:ro"

echo "Running $IMAGE with GPU support..."
docker run --rm -it \
    --gpus all \
    -v "$SAVES_DIR:/app/saves" \
    $extra_vols \
    "$IMAGE" "$@"
