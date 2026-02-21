# PVM CUDA Native Backend

Native **C++17 / CUDA** reimplementation of the PVM training loop, achieving ~1.4× speedup over the original Python implementation (`../python_implementation`) on the same hardware.

---

## Motivation & bottleneck analysis

The original Python implementation ran the inner training loop through the Python interpreter, introducing several performance bottlenecks:

### 1. Python interpreter overhead per step
Every training step went through attribute lookups, GIL acquisition and duck-typed dispatch — adding hundreds of microseconds of CPU overhead between CUDA kernel launches. At 300+ steps/second the GPU frequently stalled waiting for the host.

### 2. No CUDA Graphs
The Python version launched CUDA kernels individually from Python in a loop. Each launch incurs a host→driver round-trip. CUDA Graphs capture the *entire* training step (forward + backward + weight update) as a single replayable graph; subsequent steps replay with zero host-side kernel-launch overhead.

### 3. Synchronous data pipeline
The Python data loader (zip extraction → JPEG decode via OpenCV/numpy → type conversion → HtoD copy) ran synchronously on the main thread, stalling the GPU between steps. The C++ backend uploads frames over a **dedicated CUDA stream** with **pinned host memory** (`cudaMallocHost`), overlapping upload of batch `n+1` with compute of batch `n`.

### 4. Memory allocation per step
Python's numpy pipeline allocated temporary arrays every step (resize, cvtColor, float conversion). C++ reuses a single pre-allocated pinned staging buffer for the lifetime of training.

---

## What is accelerated

| Component | Python | C++ native |
|-----------|--------|------------|
| Training loop | Python `for` + GIL | Compiled C++ |
| CUDA kernel dispatch | Python → CUDA driver | Direct C++ → CUDA driver |
| CUDA Graph | ✗ | ✓ (captured after 10 warm-up steps) |
| Frame upload | Synchronous, numpy allocs | Async stream + pinned memory |
| Data loading | Python zipfile + numpy | C++ unzip + OpenCV `imdecode` |
| Batching | 1 sequence | N independent sequences |

---

## Performance

**Test hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, compute capability 8.9), driver 590.48.01 / CUDA 13.1

| | Python implementation | C++ native backend |
|---|---|---|
| ETA (100M steps, `small.json`) | ~130 h | ~92 h |
| Speedup | baseline | **~1.4×** |

---

## Project structure

```
cuda_native_backend/
├── include/
│   ├── pvm_config.h          # JSON config → PVMConfig struct
│   ├── pvm_graph.h           # PVMUnit, PVMGraph — connectivity graph
│   ├── pvm_kernels.cuh       # All CUDA kernel declarations
│   ├── pvm_object.cuh        # PVMObject class (main training object)
│   ├── data_provider.h       # Abstract DataProvider + implementations
│   └── training_manager.h    # TrainingManager class
├── src/
│   ├── pvm_graph.cpp         # Graph generation
│   ├── pvm_kernels.cu        # CUDA kernels (forward, backward)
│   ├── pvm_object.cu         # PVMObject: alloc, forward, backward, CUDA Graph
│   ├── data_provider.cpp     # Zip/Video/ImageDir data providers
│   ├── training_manager.cpp  # Training loop with progress display
│   └── main.cpp              # CLI entry point
├── third_party/nlohmann/json.hpp  # Auto-downloaded by docker_build.sh
├── CMakeLists.txt
├── Dockerfile
├── docker_build.sh           # Fetch dependencies + build Docker image
├── docker_run.sh             # Run training in Docker (--shell for interactive)
└── entrypoint.sh
```

---

## Quick start (Docker)

```bash
# From the cuda_native_backend/ directory:
./docker_build.sh

# Train on a zip file
./docker_run.sh \
    -S /model_zoo/small.json \
    -f /pvm_data/green_ball_long.pkl.zip

# Interactive shell inside container
./docker_run.sh --shell
# then: /pvm_cpp/build/pvm -S /model_zoo/small.json -f /pvm_data/green_ball_long.pkl.zip

# Load checkpoint and continue
./docker_run.sh \
    -S /model_zoo/small.json \
    -f /pvm_data/green_ball_long.pkl.zip \
    -L /saves/pvm_save_model_000100000.bin
```

The script mounts `../PVM_data` → `/pvm_data` and `../python_implementation/model_zoo` → `/model_zoo` automatically.

---

## Local build (requires CUDA toolkit + OpenCV)

```bash
cd cuda_native_backend
mkdir -p third_party/nlohmann
curl -fsSL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp \
     -o third_party/nlohmann/json.hpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/pvm -S ../python_implementation/model_zoo/small.json \
            -f ../PVM_data/green_ball_long.pkl.zip
```

---

## CLI reference

```
pvm -S <spec.json> [options]
  -S, --spec    <file>   Model JSON spec          (required)
  -L, --load    <file>   Load checkpoint
  -d, --dataset <name>   Dataset subfolder name
  -p, --path    <dir>    Base data path
  -f, --file    <file>   Direct video/zip/image-dir path
  -b, --batch   <N>      Batch size (default: 1)
  -G, --no-graph         Disable CUDA Graph
```

## Progress output
```
12500000/100000000 (12.5%) | fps: 318.2 avg / 321.4 inst | ETA: 77h12m04s
```

## Model zoo compatibility

All `python_implementation/model_zoo/*.json` files work directly.

---

## License

Apache 2.0.  
Original Python implementation © 2023 Filip Piekniewski.  
C++ native backend © 2026 Radosław Szewczyk.
