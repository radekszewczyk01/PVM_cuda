#!/usr/bin/env bash
set -euo pipefail

# Allow dropping into a shell for debugging:
#   docker run ... pvm_cuda_cpp bash
if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
    exec "${1}" "${@:2}"
fi

# The pvm binary is at /pvm_cpp/build/pvm after Docker build.
exec /pvm_cpp/build/pvm "$@"
