FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 AS pvm_base

ENV DISPLAY=:1.0
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y \
     vim \
     git \
     python3-pip \
     python3-opencv \
     netcat \
     telnet \
     make \
  && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip3 install "numpy<2" pycuda && \
    grep -rL 'from __future__ import annotations' \
        /usr/local/lib/python3.10/dist-packages/pycuda/compyte/ | \
    xargs -I{} sed -i '1i from __future__ import annotations' {}

# Copy source into image to pre-build libpvm.so during docker build.
# This avoids a potentially long compilation on every container start.
# At runtime, the host dir is volume-mounted over /pvm, so the entrypoint
# copies the pre-built libpvm.so into place if the source hasn't changed.
COPY . /pvm_build
RUN make -C /pvm_build/pvmcuda_pkg/backend_c clean && \
    make -C /pvm_build/pvmcuda_pkg/backend_c

COPY entrypoint.sh /
WORKDIR /pvm
ENTRYPOINT [ "/entrypoint.sh" ]
