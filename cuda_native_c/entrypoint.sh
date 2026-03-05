#!/usr/bin/env bash
# entrypoint.sh – Docker entrypoint for PVM CUDA C
set -euo pipefail

BIN="/app/build/pvm_c"

if [ ! -f "$BIN" ]; then
    echo "Binary not found: $BIN (running build inside container...)"
    cd /app && make deps && make -j"$(nproc)"
fi

exec "$BIN" "$@"
