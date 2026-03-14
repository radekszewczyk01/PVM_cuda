#!/usr/bin/env bash
set -e

cd /pvm

## Install the Python package (editable install from volume-mounted source)
pip3 install -e . --quiet 2>/dev/null

## Build C/CUDA backend.
## The Dockerfile pre-built libpvm.so into /pvm_build.  If the volume-mounted
## source at /pvm doesn't have a libpvm.so yet, copy the pre-built one.
## Then run make, which is a no-op if sources haven't changed.
BACKEND_DIR=/pvm/pvmcuda_pkg/backend_c
PREBUILD_DIR=/pvm_build/pvmcuda_pkg/backend_c

if command -v nvcc &> /dev/null; then
    # Copy the pre-built .so if it doesn't exist in the mounted volume
    if [ ! -f "$BACKEND_DIR/libpvm.so" ] && [ -f "$PREBUILD_DIR/libpvm.so" ]; then
        echo "[PVM] Copying pre-built libpvm.so from image..."
        cp "$PREBUILD_DIR/libpvm.so" "$BACKEND_DIR/libpvm.so"
    fi

    # Rebuild if sources are newer than the .so
    echo "[PVM] Building C/CUDA backend (libpvm.so)..."
    make -C "$BACKEND_DIR" 2>&1 || \
        echo "[PVM] Warning: C/CUDA backend build failed. Use -k python to run with Python backend."
else
    echo "[PVM] nvcc not found, skipping C/CUDA backend build."
fi

## Running passed command
if [[ "$1" ]]; then
    eval "$@"
fi
