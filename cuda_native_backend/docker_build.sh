#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Fetch nlohmann/json single-header if not present ─────────────────────────
JSON_HDR="third_party/nlohmann/json.hpp"
if [[ ! -f "$JSON_HDR" ]]; then
    echo "[build] Fetching nlohmann/json single-header..."
    mkdir -p third_party/nlohmann
    curl -fsSL \
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" \
        -o "$JSON_HDR"
    echo "[build] Done → $JSON_HDR"
fi

# ── Docker image name ─────────────────────────────────────────────────────────
IMAGE_NAME="${IMAGE_NAME:-pvm_cuda_cpp}"

echo "[build] Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .
echo "[build] Image ready: $IMAGE_NAME"
