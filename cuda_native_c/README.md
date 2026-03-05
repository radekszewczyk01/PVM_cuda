# PVM CUDA C

Pure **CUDA C** (no C++) rewrite of the PVM predictive vision model trainer.
Eliminates all C++ STL, classes, exceptions, and third-party C++ libraries while
preserving CUDA Streams, CUDA Graphs, and cuBLAS.

## Requirements

| Tool | Notes |
|------|-------|
| NVIDIA CUDA toolkit ≥ 11.0 | provides `nvcc`, cuBLAS, CUDA runtime |
| `nvidia-smi` | GPU architecture auto-detection |
| `unzip` | required for ZIP dataset mode |
| `ffmpeg` | required for video dataset mode (with `-DPVM_VIDEO`) |

## Quick start

```bash
# 1. Download stb_image.h (one-time setup)
make deps

# 2. Build
make          # GPU arch auto-detected via nvidia-smi
# or override:
make GPU=sm_86

# 3. Train
build/pvm_c -S ../python_implementation/model_zoo/small.json \
            -f /path/to/dataset \
            -o saves/run1
```

## Build options

```
make              build binary (auto-detect GPU arch)
make GPU=sm_xx    build for specific arch (e.g. sm_86, sm_89)
make deps         download stb_image.h into third_party/stb/
make clean        remove build artefacts
make info         print detected toolchain / GPU arch
```

## Runtime options

```
Usage: build/pvm_c -S <spec.json> [options]

Required:
  -S, --spec    <file>  Model JSON spec (see python_implementation/model_zoo/)

Optional:
  -L, --load    <file>  Load checkpoint (.bin) and continue training
  -d, --dataset <name>  Dataset subdirectory name
  -p, --path    <dir>   Base data path (dataset lives under path/dataset/)
  -f, --file    <path>  Direct path to image dir / zip archive / video file
  -b, --batch   <N>     Batch size (default: 1)
  -G, --no-graph        Disable CUDA Graph capture (run eager kernels)
  -s, --save-every <N>  Steps between checkpoint saves (default: 100000)
  -o, --out     <pfx>   Checkpoint file prefix (default: pvm_save)
  -h, --help            Show this help
```

## Architecture

```
cuda_native_c/
├── Makefile               build system
├── build.sh               convenience wrapper around make
├── run.sh                 sample run command
├── Dockerfile             CUDA 12.3 + Ubuntu 22.04 image
├── include/
│   ├── pvm_config.h       PVMConfig struct + JSON parser interface
│   ├── pvm_graph.h        PVMGraph / PVMUnit structs
│   ├── pvm_kernels.h      CUDA kernel declarations
│   ├── pvm_object.h       PVMObject (GPU training state)
│   ├── data_provider.h    DataProvider polymorphic interface
│   └── training_manager.h TrainingManager
├── src/
│   ├── pvm_config.c       Hand-rolled JSON parser (no external libs)
│   ├── pvm_graph.c        Graph construction (mirrors sequence_learner.py)
│   ├── pvm_kernels.cu     14 CUDA kernels (forward / backward / data-flow)
│   ├── pvm_object.cu      GPU allocation, CUDA Graph capture, checkpoint I/O
│   ├── data_provider.c    Image dir / ZIP / synthetic backends (stb_image)
│   ├── training_manager.c Training loop with POSIX timing + FPS stats
│   └── main.c             Entry point (getopt_long argument parsing)
└── third_party/stb/
    └── stb_image.h        Single-header JPEG/PNG decoder (make deps)
```

## Key design decisions

- **CUDA Streams** – two streams: `stream_compute` and `stream_upload`. Frame
  upload overlaps computation of the previous step via `cudaStreamWaitEvent`.
- **CUDA Graphs** – after `warmup_steps` eager steps the forward+backward pass
  is captured into a `cudaGraph_t` and replayed with `cudaGraphLaunch`. The
  graph is invalidated and re-captured whenever the learning rate changes.
- **cuBLAS** – `cublasSscal` / `cublasSaxpy` for error computation (sign + add).
  One global handle with reference counting.
- **No C++** – all `.cu` files expose only C-callable (`extern "C"`) symbols.
  Kernels compile as CUDA but host interface is plain C.
- **JSON** – minimal hand-rolled parser in `pvm_config.c`; covers the flat PVM
  spec format without any external library.
- **Image loading** – `stb_image.h` + inline bilinear resize (no OpenCV, no
  stb_image_resize).

## Video support

Compile with `-DPVM_VIDEO` to enable the `ffmpeg` video backend:

```bash
make EXTRA_FLAGS=-DPVM_VIDEO
```

(add `EXTRA_FLAGS` to `C_FLAGS` in the Makefile or export it).

## Checkpoint format

Binary format, little-endian:
```
magic   : uint32  0x50564D43 ('PVMC')
step    : int64
n_weights: int32
weights : float32 × n_weights
```
