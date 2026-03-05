#!/usr/bin/env bash
# run.sh – Run PVM CUDA C training
# All arguments are passed directly to build/pvm_c
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

BIN="$SCRIPT_DIR/build/pvm_c"

if [ ! -f "$BIN" ]; then
    echo "Binary not found: $BIN"
    echo "Run ./build.sh first"
    exit 1
fi

# Default: if no args given, show help
if [ $# -eq 0 ]; then
    exec "$BIN" --help
fi

exec "$BIN" "$@"
