# PVM CUDA — Optimized Fork

Fork of [piekniewski/PVM_cuda](https://github.com/piekniewski/PVM_cuda) — a GPU implementation of the
**Predictive Vision Model** (PVM) from:

> Piekniewski et al., *"Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network"*, 2016

---

## Repository structure

```
PVM_cuda/
├── python_implementation/    # Original Python/CUDA implementation by Filip Piekniewski
│   ├── pvmcuda_pkg/          # Python package (sequence learner, training manager, …)
│   ├── model_zoo/            # JSON model specs (small, medium, large, …)
│   ├── Dockerfile
│   └── setup.py
├── cuda_native_backend/      # Native C++/CUDA reimplementation (this work)
│   ├── src/
│   ├── include/
│   └── Dockerfile
└── PVM_data/                 # Training data (not tracked in git — download separately)
```

---

## Performance comparison

**Test hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, compute capability 8.9), driver 590.48.01 / CUDA 13.1

| Metric | Python implementation | C++ CUDA backend |
|---|---|---|
| Measured inst. fps (`small.json`) | ~215 fps | ~301 fps |
| Estimated ETA (100M steps) | ~129 h | ~92 h |
| Speedup | baseline | **~1.4×** |
| CUDA Graphs | ✗ | ✓ |
| Async data upload | ✗ | ✓ (pinned mem + stream) |

Wyniki zmierzone na tym samym zestawie danych (`green_ball_long.pkl.zip`, model `small.json`, batch size 1).

### Co oznacza output postępu?

```
6915/100000000 (0.0%) | fps: 0.5 avg / 301.4 inst | ETA: 92h08m39s
```

- **`6915/100000000`** — bieżący krok / łączna liczba kroków
- **`fps: 0.5 avg`** — średnie fps od początku trenowania (niska wartość na początku bo liczy od t=0, zanim GPU się rozgrzeje)
- **`fps: 301.4 inst`** — chwilowe fps z ostatniej sekundy — to jest **właściwa miara wydajności**
- **`ETA`** — szacowany czas do końca na podstawie chwilowego fps

> Średnie fps na początku jest mylące (GPU musi zbudować CUDA Graph i rozgrzać pamięć podręczną). Patrz na **inst fps** jako rzeczywistą metrykę.

---

## Quick start

### Python (original)
```bash
cd python_implementation/
./docker_build.sh
./docker_run.sh
# inside container:
pvm -S /pvm/model_zoo/small.json -d green_ball_training
```

### C++ native backend
```bash
cd cuda_native_backend/
./docker_build.sh
./docker_run.sh -S /model_zoo/small.json -f /pvm_data/green_ball_long.pkl.zip
```

---

## Running on a university GPU server

These instructions assume the server has Docker + NVIDIA Container Toolkit installed and
you have no write access outside of Docker (typical HPC / shared GPU setup).

### Step 0 — log in and verify GPU

```bash
ssh <user>@<server>
nvidia-smi          # check GPU model and CUDA version
docker images       # see what base images are already available
```

---

### Pure C + CUDA (cuda_native_c) — fastest

```bash
# 1. Spin up a cuda devel container (reuse one the server already has)
docker run --rm -it --gpus all \
    -v "$HOME/pvm_saves:/saves" \
    nvidia/cuda:11.8.0-devel-ubuntu22.04 bash

# 2. Inside the container — install deps and clone
apt-get update && apt-get install -y git make curl unzip ffmpeg
git clone https://github.com/radekszewczyk01/PVM_cuda /PVM_cuda
cd /PVM_cuda/cuda_native_c

# 3. Build (auto-detects GPU arch)
make deps && make -j$(nproc)

# 4. Train (synthetic data — no dataset needed)
build/pvm_c -S ../python_implementation/model_zoo/small.json -o /saves/run_c

# 4b. Train with a real dataset (zip or image directory)
build/pvm_c -S ../python_implementation/model_zoo/small.json \
            -f /path/to/dataset.zip -o /saves/run_c
```

Checkpoints are written to `~/pvm_saves/` on the host and persist after the container exits.
Press **Ctrl+C** to stop — a final checkpoint is saved automatically.

---

### C++ CUDA backend (cuda_native_backend)

```bash
docker run --rm -it --gpus all \
    -v "$HOME/pvm_saves:/saves" \
    nvidia/cuda:11.8.0-devel-ubuntu22.04 bash

# inside:
apt-get update && apt-get install -y git cmake libopencv-dev
git clone https://github.com/radekszewczyk01/PVM_cuda /PVM_cuda
cd /PVM_cuda/cuda_native_backend
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# run (requires a data file — no synthetic fallback)
# create a dummy image dir if you have no dataset:
apt-get install -y imagemagick && mkdir /tmp/data
convert -size 64x64 xc:gray /tmp/data/frame0001.jpg
./pvm -S ../../python_implementation/model_zoo/small.json -f /tmp/data
```

---

### Python implementation (pvmcuda_pkg)

```bash
docker run --rm -it --gpus all \
    -v "$HOME/pvm_saves:/saves" \
    nvidia/cuda:11.8.0-devel-ubuntu22.04 bash

# inside:
apt-get update && apt-get install -y git python3 python3-pip
git clone https://github.com/radekszewczyk01/PVM_cuda /PVM_cuda
cd /PVM_cuda/python_implementation

pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip3 install pycuda cupy-cuda11x opencv-python-headless numpy
pip3 install -e .

# run with synthetic data (-s) and skip readout model (-r)
python3 -m pvmcuda_pkg.run -S model_zoo/small.json -s -r

# run with a real zip dataset
python3 -m pvmcuda_pkg.run -S model_zoo/small.json \
        -f /path/to/dataset.zip -r
```

---

### Performance comparison (RTX 5000 Ada, sm_89, synthetic data, small.json)

| Implementation | inst. FPS | ETA (100M steps) |
|---|---|---|
| Pure C + CUDA Graphs (`cuda_native_c`) | ~650 | ~43 h |
| C++ + CUDA Graphs (`cuda_native_backend`) | ~715 | ~39 h |
| Python + pycuda (`pvmcuda_pkg`) | TBD | TBD |

> C++ is slightly faster than pure C on the server GPUs because of more aggressive compiler optimisations in g++ vs nvcc `--x c` mode. The difference is small and both are far ahead of Python.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).  
Original work © 2023 Filip Piekniewski.  
CUDA native backend © 2026 Radosław Szewczyk.
