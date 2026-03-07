FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# cuBLAS .so files live in the devel image under /usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    unzip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the repo root (see docker_build.sh)
COPY cuda_native_c/ .
COPY python_implementation/model_zoo/ /app/model_zoo/

# Download stb_image.h and build from source (clean to avoid reusing host .o files)
# GPU arch is passed via --build-arg GPU=sm_XX (see docker_build.sh)
# RTX 5000 Ada Generation = sm_89
ARG GPU=sm_89
RUN make deps && make clean && make -j$(nproc) GPU=${GPU}

ENTRYPOINT ["./entrypoint.sh"]
