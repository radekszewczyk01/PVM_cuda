#!/usr/bin/env bash
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
IMAGE_NAME="${IMAGE_NAME:-pvm_cuda_cpp}"
DATA_DIR="${DATA_DIR:-"$(pwd)/../PVM_data"}"
MODEL_ZOO="${MODEL_ZOO:-"$(pwd)/../python_implementation/model_zoo"}"
SAVE_DIR="${SAVE_DIR:-"$(pwd)/saves"}"
CONTAINER_NAME="${CONTAINER_NAME:-pvm_cpp_run}"

mkdir -p "$SAVE_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
# Pass --shell as first argument to get an interactive bash session instead.
if [[ "${1:-}" == "--shell" ]]; then
    shift
    docker run --rm -it \
        --name "${CONTAINER_NAME}_shell" \
        --gpus all \
        -v "$(realpath "$DATA_DIR"):/pvm_data:ro" \
        -v "$(realpath "$MODEL_ZOO"):/model_zoo:ro" \
        -v "$(realpath "$SAVE_DIR"):/saves" \
        "$IMAGE_NAME" bash
else
    docker run --rm \
        --name "$CONTAINER_NAME" \
        --gpus all \
        -v "$(realpath "$DATA_DIR"):/pvm_data:ro" \
        -v "$(realpath "$MODEL_ZOO"):/model_zoo:ro" \
        -v "$(realpath "$SAVE_DIR"):/saves" \
        "$IMAGE_NAME" \
        "$@"
fi
