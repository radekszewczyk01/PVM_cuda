// PVM CUDA Kernels
// Optimized vs original Python pycuda kernels:
//   - tile-shared memory for dot products
//   - fused sigmoid+bias in single kernel
//   - better memory coalescing

#include "pvm_kernels.cuh"
#include <cuda_runtime.h>

#define TILE 32

// ─────────────────────────────────────────────────────────────────────────────
// Forward: batched dot product  W[shape0 x shape1] * x[shape1] -> y[shape0]
// One thread per (unit, input_col).  Uses atomicAdd into output.
// Same as original gpu_dot_fast_set_bias but with improved memory access.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_dot_fast_set_bias(
    const float* __restrict__ W,
    const float* __restrict__ x,
          float* __restrict__ y,
    const int*   w_ptr,
    const int*   x_ptr,
    const int*   y_ptr,
    const int*   shape0,   // output dim
    const int*   shape1,   // input dim (columns)
    const int*   obj_id,
    const int*   col_id,
    int total_threads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    int unit = obj_id[tid];
    int col  = col_id[tid];

    const float* w = W + w_ptr[unit];
    const float  v = x[x_ptr[unit] + col];
    float*       r = y + y_ptr[unit];
    int          m = shape0[unit];

    // Each col contributes to all output rows via W[row, col]
    for (int row = 0; row < m; ++row)
        atomicAdd(&r[row], w[row * shape1[unit] + col] * v);

    // bias: last column (col == shape1[unit]-1) sets bias node to 1.0
    if (col == shape1[unit] - 1)
        r[m] = 1.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Activated sigmoids (in-place)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_sigmoid_fast(
    float* __restrict__ mem,
    const int*   ptr,
    const float* beta,
    const int*   shape0,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    float* r = mem + ptr[i];
    float  b = beta[i];
    for (int k = 0; k < shape0[i]; ++k)
        r[k] = 1.0f / (1.0f + expf(-b * r[k]));
}

__global__ void k_sigmoid_poly_fast(
    float* __restrict__ mem,
    const int*   ptr,
    const float* beta,
    const int*   shape0,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    float* r = mem + ptr[i];
    for (int k = 0; k < shape0[i]; ++k) {
        float v = r[k];
        r[k] = (v / (2.0f * (fabsf(v) + 1.0f))) + 0.5f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sign function (for abs-diff error)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_sgn(float* mem, int total_obj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    mem[i] = copysignf(1.0f, mem[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward: W^T * delta  (build intermediate matrix for later reduction)
// Each thread handles one (unit, col) of W and multiplies a row of delta.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_dot_transpose_fast(
    const float* __restrict__ W,
          float* __restrict__ W_buf,
    const float* __restrict__ delta,
    const int* w_ptr,
    const int* d_ptr,
    const int* shape0,
    const int* shape1,
    const int* obj_id,
    const int* col_id,
    int total_threads)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    int unit = obj_id[tid];
    int col  = col_id[tid];

    const float* w  = W     + w_ptr[unit];
          float* wb = W_buf + w_ptr[unit];
    const float* d  = delta + d_ptr[unit];
    int m0 = shape0[unit];

    for (int row = 0; row < m0; ++row)
        wb[row * shape1[unit] + col] = w[row * shape1[unit] + col] * d[row];
}

// ─────────────────────────────────────────────────────────────────────────────
// Sum W_buf rows to build error vector (column sum after k_dot_transpose_fast)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_sum_dot_transpose(
    const float* __restrict__ W_buf,
          float* __restrict__ error,
    const int* w_ptr,
    const int* e_ptr,
    const int* shape0,
    const int* shape1,
    const int* obj_id,
    const int* col_id,
    int total_threads)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    int unit = obj_id[tid];
    int col  = col_id[tid];

    const float* wb = W_buf + w_ptr[unit];
    float*       e  = error + e_ptr[unit];
    int m0 = shape0[unit];

    float s = 0.f;
    for (int row = 0; row < m0; ++row)
        s += wb[row * shape1[unit] + col];
    atomicAdd(&e[col], s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sigmoid derivative:  delta = activ * (1 - activ) * error
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_sigmoid_der_mul(
    const float* __restrict__ activ,
    const float* __restrict__ error,
          float* __restrict__ delta,
    const int* a_ptr,
    const int* e_ptr,
    const int* d_ptr,
    const int* shape0,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    const float* a = activ + a_ptr[i];
    const float* e = error + e_ptr[i];
    float*       d = delta + d_ptr[i];
    for (int k = 0; k < shape0[i]; ++k)
        d[k] = a[k] * (1.0f - a[k]) * e[k];
}

// ─────────────────────────────────────────────────────────────────────────────
// Rational sigmoid derivative:  delta = (1 / (2*(|x|+1)^2)) * error
// where x is the pre-activation value recovered from output a via inverse
// Note: for poly sigmoid a = x/(2(|x|+1)) + 0.5 => x = (2a-1)/(1±(2a-1))
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_sigmoid_poly_der_mul(
    const float* __restrict__ activ,
    const float* __restrict__ error,
          float* __restrict__ delta,
    const int* a_ptr,
    const int* e_ptr,
    const int* d_ptr,
    const int* shape0,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    const float* a = activ + a_ptr[i];
    const float* e = error + e_ptr[i];
    float*       d = delta + d_ptr[i];
    for (int k = 0; k < shape0[i]; ++k) {
        float m  = copysignf(1.0f, a[k] - 0.5f);
        float xv = fmaf(2.0f, a[k], -1.0f) / (1.0f + m * fmaf(2.0f, a[k], -1.0f));
        float xp  = fabsf(xv) + 1.0f;
        d[k] = (1.0f / (2.0f * xp * xp)) * e[k];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight update with momentum:
//   dW_new[l,k] = momentum * dW_old[l,k] + lr * delta[l] * input[k]
//   W += dW_new
// One thread handles one (unit, input_col) row of the outer product
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_outer_update(
    const float* __restrict__ delta,
    const float* __restrict__ input,
    const float* __restrict__ dW_prev,
          float* __restrict__ dW,
    const int*   d_ptr,
    const int*   i_ptr,
    const int*   dW_ptr,
    const int*   shape0,
    const int*   shape1,
    const float* lr,
    const float* mom,
    const int*   obj_id,
    const int*   col_id,
    int total_threads)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    int unit = obj_id[tid];
    int col  = col_id[tid];

    const float* del  = delta   + d_ptr[unit];
    const float  inp  = input[i_ptr[unit] + col];
    const float* dp   = dW_prev + dW_ptr[unit];
    float*       dw   = dW      + dW_ptr[unit];
    float alpha  = lr [unit];
    float beta_m = mom[unit];
    int   m0     = shape0[unit];
    int   m1     = shape1[unit];

    for (int row = 0; row < m0; ++row) {
        int idx = row * m1 + col;
        dw[idx] = fmaf(beta_m, dp[idx], alpha * del[row] * inp);
    }
}

// W += dW (fused for all weights)
__global__ void k_weight_add(float* __restrict__ W, const float* __restrict__ dW, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    W[i] += dW[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Data flow: copy variable-length blocks between flat GPU arrays
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_copy_blocks(
    const float* __restrict__ from,
          float* __restrict__ to,
    const int* from_ptr,
    const int* from_qnt,
    const int* to_ptr,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int n = from_qnt[i];
    const float* f = from + from_ptr[i];
    float*       t = to   + to_ptr[i];
    for (int j = 0; j < n; ++j) t[j] = f[j];
}

__global__ void k_copy_blocks_comp(
    const float* __restrict__ from,
          float* __restrict__ to,
    const int* from_ptr,
    const int* from_qnt,
    const int* to_ptr,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int n = from_qnt[i];
    const float* f = from + from_ptr[i];
    float*       t = to   + to_ptr[i];
    for (int j = 0; j < n; ++j)
        t[j] = 0.8f * f[j] + 0.1f;  // compress to (0.1, 0.9)
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame distribution: split frame into patches for layer-0 units
// frame: [H, W, 3] float32 (row-major)
// Each thread handles one unit patch
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_dist_frame(
    const float* __restrict__ frame,
          float* __restrict__ input_mem,
    const int* unit_ptr,
    int shape0, int shape1,    // total frame dims (H, W)
    int dx, int dy,            // number of unit tiles in x,y
    int sx, int sy,            // tile size in pixels
    int input_offset,          // offset within unit input buffer
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int x_block = i / dy, y_block = i % dy;
    int x0 = x_block * sx, y0 = y_block * sy;
    float* mem = input_mem + unit_ptr[i] + input_offset;

    for (int j = 0; j < sy; ++j)
    for (int k = 0; k < sx; ++k) {
        int x = x0 + k, y = y0 + j;
        int frame_idx = 3 * (shape0 * x + y);
        int mem_idx   = 3 * (k * sy + j);
        mem[mem_idx    ] = frame[frame_idx    ];
        mem[mem_idx + 1] = frame[frame_idx + 1];
        mem[mem_idx + 2] = frame[frame_idx + 2];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Collect prediction frame from unit output buffers
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_collect_frame(
          float* __restrict__ frame,
    const float* __restrict__ output_mem,
    const int* unit_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int input_offset,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int x_block = i / dy, y_block = i % dy;
    int x0 = x_block * sx, y0 = y_block * sy;
    const float* mem = output_mem + unit_ptr[i] + input_offset;

    for (int j = 0; j < sy; ++j)
    for (int k = 0; k < sx; ++k) {
        int x = x0 + k, y = y0 + j;
        int frame_idx = 3 * (shape0 * x + y);
        int mem_idx   = 3 * (k * sy + j);
        frame[frame_idx    ] = mem[mem_idx    ];
        frame[frame_idx + 1] = mem[mem_idx + 1];
        frame[frame_idx + 2] = mem[mem_idx + 2];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Collect layer activation into uint8 image
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_collect_activ(
    unsigned int* __restrict__ frame,
    const float*  __restrict__ repr_mem,
    const int*    repr_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int ptr_offset,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int x_block = i / dy, y_block = i % dy;
    int x0 = x_block * sx, y0 = y_block * sy;
    const float* mem = repr_mem + repr_ptr[i + ptr_offset];

    for (int j = 0; j < sy; ++j)
    for (int k = 0; k < sx; ++k) {
        int x = x0 + k, y = y0 + j;
        int frame_idx = shape0 * x + y;
        frame[frame_idx] = min(255u, (unsigned int)(255.0f * mem[k * sy + j]));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Copy variable-size repr blocks (for context in complex layer mode)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_copy_repr_blocks(
    const float* __restrict__ from,
          float* __restrict__ to,
    const int* from_ptr,
    const int* to_ptr,
    const int* size,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    const float* f = from + from_ptr[i];
    float*       t = to   + to_ptr[i];
    for (int j = 0; j < size[i]; ++j) t[j] = f[j];
}
