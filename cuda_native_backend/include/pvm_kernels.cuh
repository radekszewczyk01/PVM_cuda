#pragma once
// CUDA kernel declarations for PVM
// All kernels mirror the original Python pycuda SourceModule kernels

#include <cuda_runtime.h>
#include <cublas_v2.h>

// ───────── Forward pass kernels ─────────────────────────────────────────────

// dot product + bias, parallel over input rows (same as gpu_dot_fast_set_bias)
__global__ void k_dot_fast_set_bias(
    const float* __restrict__ weights,
    const float* __restrict__ input,
          float* __restrict__ output,
    const int*   w_ptr,
    const int*   in_ptr,
    const int*   out_ptr,
    const int*   shape0,   // output dim (rows)
    const int*   shape1,   // input dim  (cols)
    const int*   obj_id,
    const int*   row_id,
    int total_threads
);

// sigmoid activation in-place
__global__ void k_sigmoid_fast(
    float* __restrict__ mem,
    const int* ptr,
    const float* beta,
    const int*   shape0,
    int total_obj
);

// rational polynomial sigmoid activation in-place (faster than exp)
__global__ void k_sigmoid_poly_fast(
    float* __restrict__ mem,
    const int* ptr,
    const float* beta,
    const int*   shape0,
    int total_obj
);

// elem-wise sign (for abs-diff error)
__global__ void k_sgn(float* mem, int total_obj);

// ───────── Backward pass kernels ────────────────────────────────────────────

// W^T * delta (builds intermediate weighted matrix)
__global__ void k_dot_transpose_fast(
    const float* __restrict__ W,
          float* __restrict__ W_buf,
    const float* __restrict__ delta,
    const int* w_ptr,
    const int* delta_ptr,
    const int* shape0,
    const int* shape1,
    const int* obj_id,
    const int* row_id,
    int total_threads
);

// sum of W^T * delta along rows -> error vector
__global__ void k_sum_dot_transpose(
    const float* __restrict__ W_buf,
          float* __restrict__ error,
    const int* w_ptr,
    const int* err_ptr,
    const int* shape0,
    const int* shape1,
    const int* obj_id,
    const int* row_id,
    int total_threads
);

// sigmoid derivative * error -> delta   (delta = a*(1-a)*error)
__global__ void k_sigmoid_der_mul(
    const float* __restrict__ activ,
    const float* __restrict__ error,
          float* __restrict__ delta,
    const int* a_ptr,
    const int* e_ptr,
    const int* d_ptr,
    const int* shape0,
    int total_obj
);

// rational sigmoid derivative * error -> delta
__global__ void k_sigmoid_poly_der_mul(
    const float* __restrict__ activ,
    const float* __restrict__ error,
          float* __restrict__ delta,
    const int* a_ptr,
    const int* e_ptr,
    const int* d_ptr,
    const int* shape0,
    int total_obj
);

// weight update:  dW = momentum * dW_prev + lr * delta x input   (outer product with momentum)
__global__ void k_outer_update(
    const float* __restrict__ delta,    // upper delta
    const float* __restrict__ input,    // lower activation
    const float* __restrict__ dW_prev,  // previous momentum buffer
          float* __restrict__ dW,       // result weight change (also new momentum)
    const int* delta_ptr,
    const int* input_ptr,
    const int* dW_ptr,
    const int* shape0,
    const int* shape1,
    const float* lr,
    const float* momentum,
    const int*   obj_id,
    const int*   row_id,
    int total_threads
);

// W += dW
__global__ void k_weight_add(float* W, const float* dW, int total_weights);

// ───────── Data flow kernels ─────────────────────────────────────────────────

// Copy blocks of memory (context/primary flow between units)
__global__ void k_copy_blocks(
    const float* __restrict__ from,
          float* __restrict__ to,
    const int* from_ptr,
    const int* from_qnt,
    const int* to_ptr,
    int total_obj
);

// Copy blocks with linear compression to (0.1, 0.9)
__global__ void k_copy_blocks_comp(
    const float* __restrict__ from,
          float* __restrict__ to,
    const int* from_ptr,
    const int* from_qnt,
    const int* to_ptr,
    int total_obj
);

// Distribute frame patches to layer-0 unit input buffers  (3 channels)
__global__ void k_dist_frame(
    const float* __restrict__ frame,
          float* __restrict__ input_mem,
    const int* unit_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int input_offset,
    int total_obj
);

// Collect prediction from output buffer into a frame (3 channels)
__global__ void k_collect_frame(
          float* __restrict__ frame,
    const float* __restrict__ output_mem,
    const int* unit_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int input_offset,
    int total_obj
);

// Collect layer activation into uint8 image buffer (for visualization)
__global__ void k_collect_activ(
    unsigned int* __restrict__ frame,
    const float*  __restrict__ repr_mem,
    const int* repr_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int ptr_offset,
    int total_obj
);

// Copy repr blocks (for context feeding)
__global__ void k_copy_repr_blocks(
    const float* __restrict__ from,
          float* __restrict__ to,
    const int* from_ptr,
    const int* to_ptr,
    const int* size,
    int total_obj
);
