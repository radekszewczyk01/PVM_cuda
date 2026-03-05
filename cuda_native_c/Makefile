# Makefile – PVM CUDA C
# Builds the PVM predictive vision model (pure CUDA C, no C++).
#
# Requirements:
#   - NVIDIA CUDA toolkit (nvcc)
#   - cuBLAS (included with CUDA)
#   - stb_image.h in third_party/stb/ (run: make deps)
#
# Usage:
#   make deps       – download stb_image.h
#   make            – build binary
#   make clean      – remove build artefacts
#   make GPU=sm_86  – override GPU architecture (default: detected)

# ── Toolchain ──────────────────────────────────────────────────────────────
CUDA_PATH  ?= /usr/local/cuda
NVCC       := $(CUDA_PATH)/bin/nvcc
CC         := $(CUDA_PATH)/bin/nvcc   # compile .c files via nvcc for includes

# ── GPU architecture ────────────────────────────────────────────────────────
# Auto-detect with nvidia-smi; override with GPU=<arch> on command line
ifndef GPU
GPU := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -1 | tr -d '.' | sed 's/^/sm_/')
ifeq ($(GPU),sm_)
GPU := sm_86
endif
endif
ARCH_FLAGS := -gencode arch=compute_$(subst sm_,,$(GPU)),code=$(GPU)

# ── Flags ───────────────────────────────────────────────────────────────────
INCS := -I include -I third_party/stb -I $(CUDA_PATH)/include

# Flags for CUDA C source files (.cu) – compiled as CUDA device+host C
# Note: nvcc uses g++ for host code in .cu files, so no -std=gnuXX here.
CU_FLAGS := -O3 --x cu $(ARCH_FLAGS) $(INCS) \
            --compiler-options='-O3 -Wall -Wno-unused-result' \
            -diag-suppress 1650 \
            -lineinfo

# Flags for plain C source files (.c) – compiled via nvcc host-only path
C_FLAGS  := -O3 --x c $(INCS) \
            --compiler-options='-O3 -Wall -std=gnu11 -Wno-unused-result -Wno-format-truncation'

# Linker flags
LD_LIBS  := -lcublas -lcudart -lm

# ── Directories ─────────────────────────────────────────────────────────────
BUILD := build
BIN   := $(BUILD)/pvm_c

# ── Sources ─────────────────────────────────────────────────────────────────
CU_SRCS := src/pvm_kernels.cu \
           src/pvm_object.cu

C_SRCS  := src/pvm_config.c \
           src/pvm_graph.c \
           src/data_provider.c \
           src/training_manager.c \
           src/main.c

CU_OBJS := $(patsubst src/%.cu, $(BUILD)/%.cu.o, $(CU_SRCS))
C_OBJS  := $(patsubst src/%.c,  $(BUILD)/%.c.o,  $(C_SRCS))

ALL_OBJS := $(CU_OBJS) $(C_OBJS)

# ── Targets ─────────────────────────────────────────────────────────────────
.PHONY: all clean deps info help

all: deps $(BIN)

$(BIN): $(ALL_OBJS)
	@mkdir -p $(BUILD)
	@echo "Linking $@  [GPU=$(GPU)]"
	$(NVCC) $(ARCH_FLAGS) $(ALL_OBJS) $(LD_LIBS) -o $@
	@echo "Built: $@"

# Compile CUDA C files
$(BUILD)/%.cu.o: src/%.cu | $(BUILD)
	@echo "  NVCC  $<"
	$(NVCC) $(CU_FLAGS) -c $< -o $@

# Compile plain C files (host only, routed through nvcc for CUDA includes)
$(BUILD)/%.c.o: src/%.c | $(BUILD)
	@echo "  CC    $<"
	$(NVCC) $(C_FLAGS) -c $< -o $@

$(BUILD):
	@mkdir -p $(BUILD)

# ── Dependencies: download stb_image.h ─────────────────────────────────────
STB_DIR  := third_party/stb
STB_IMG  := $(STB_DIR)/stb_image.h

deps: $(STB_IMG)

$(STB_IMG):
	@mkdir -p $(STB_DIR)
	@echo "Downloading stb_image.h from GitHub..."
	@curl -fsSL -o $(STB_IMG) \
	  https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
	  || wget -q -O $(STB_IMG) \
	  https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
	@echo "OK: $(STB_IMG)"

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD)

info:
	@echo "CUDA_PATH : $(CUDA_PATH)"
	@echo "NVCC      : $(NVCC)"
	@echo "GPU arch  : $(GPU)"
	@echo "ARCH_FLAGS: $(ARCH_FLAGS)"
	@echo "CU_SRCS   : $(CU_SRCS)"
	@echo "C_SRCS    : $(C_SRCS)"

help:
	@echo "Targets:"
	@echo "  all    – build pvm_c binary (default)"
	@echo "  deps   – download stb_image.h"
	@echo "  clean  – remove build directory"
	@echo "  info   – show build configuration"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH=<path>  (default: /usr/local/cuda)"
	@echo "  GPU=sm_XX         (default: auto-detected)"
