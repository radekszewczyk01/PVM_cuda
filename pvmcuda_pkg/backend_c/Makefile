# Makefile for libpvm.so -- C/CUDA backend for PVM
#
# Usage:
#   make            # build libpvm.so
#   make clean      # remove build artifacts

NVCC       ?= nvcc
NVCCFLAGS  = -shared -Xcompiler -fPIC -O2 -Iinclude
# GPU architecture flags.  Default builds for several common architectures.
# Override with: make ARCH="-arch=sm_86" for a specific GPU.
# Use ARCH="" to let nvcc pick the default.
ARCH       ?= -arch=sm_70 -gencode=arch=compute_70,code=sm_70 \
              -gencode=arch=compute_80,code=sm_80 \
              -gencode=arch=compute_86,code=sm_86 \
              -gencode=arch=compute_89,code=sm_89

SRCS = src/kernels.cu \
       src/pvm_model.cu \
       src/mlp.cu \
       src/readout.cu \
       src/pvm_api.cu

HDRS = $(wildcard include/*.h)

TARGET = libpvm.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(ARCH) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
