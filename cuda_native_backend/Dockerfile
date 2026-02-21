FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    unzip \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

# ── nlohmann/json single-header ───────────────────────────────────────────────
# Copied from build context; see docker_build.sh which fetches it first
WORKDIR /pvm_cpp
COPY . .

# ── Build ─────────────────────────────────────────────────────────────────────
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -G Ninja \
 && cmake --build build --parallel

# ── Runtime entrypoint ────────────────────────────────────────────────────────
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/pvm_cpp/entrypoint.sh"]
