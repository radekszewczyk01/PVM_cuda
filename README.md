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

| Metric               | Python implementation | C++ CUDA backend |
|----------------------|-----------------------|------------------|
| Estimated ETA (100M steps, `small.json`) | ~130 h | ~92 h |
| Speedup              | baseline              | **~1.4×**        |
| CUDA Graphs          | ✗                     | ✓                |
| Async data upload    | ✗                     | ✓ (pinned mem + stream) |

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

## License

Apache 2.0 — see [LICENSE](LICENSE).  
Original work © 2023 Filip Piekniewski.  
CUDA native backend © 2026 Radosław Szewczyk.
