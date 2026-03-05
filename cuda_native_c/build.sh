#!/usr/bin/env bash
# build.sh – Build PVM CUDA C
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# ── Optional: inside a Docker container, set CUDA_PATH if needed ────────────
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

if ! command -v "$CUDA_PATH/bin/nvcc" &>/dev/null; then
    echo "ERROR: nvcc not found at $CUDA_PATH/bin/nvcc"
    echo "Set CUDA_PATH to your CUDA installation."
    exit 1
fi

# ── Detect GPU arch ──────────────────────────────────────────────────────────
GPU_ARCH="${GPU:-}"
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -1 | tr -d '.' | sed 's/^/sm_/' || echo "sm_86")
    [ -z "$GPU_ARCH" ] && GPU_ARCH="sm_86"
fi
echo "GPU architecture: $GPU_ARCH"

# ── Number of parallel jobs ──────────────────────────────────────────────────
JOBS="${JOBS:-$(nproc)}"

# ── Build ─────────────────────────────────────────────────────────────────────
echo "=== Building PVM CUDA C  (jobs=$JOBS) ==="
make -j"$JOBS" GPU="$GPU_ARCH" all

echo ""
echo "=== Build complete: build/pvm_c ==="
echo "Usage:"
echo "  ./run.sh -S <model_zoo/small.json> -f <data_dir_or_zip>"
