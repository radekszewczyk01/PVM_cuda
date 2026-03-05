/* pvm_kernels.cu – CUDA kernel implementations for PVM (plain C CUDA)
 * Compiled with nvcc --x cu; no C++ runtime required.
 */
#include "pvm_kernels.h"
#include <cuda_runtime.h>
#include <math.h>

/* ── Forward: dot product ─────────────────────────────────────────────────── */
/* One thread per (unit, input_col).
 * Each col multiplies its weight column into output rows via atomicAdd.
 * The bias node (last col) is set after all partial products accumulate.  */
__global__ void k_dot_fast_set_bias(
    const float * __restrict__ W,
    const float * __restrict__ x,
          float * __restrict__ y,
    const int   *w_ptr,
    const int   *x_ptr,
    const int   *y_ptr,
    const int   *shape0,
    const int   *shape1,
    const int   *obj_id,
    const int   *col_id,
    int total_threads)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    int unit = obj_id[tid];
    int col  = col_id[tid];

    const float *w = W + w_ptr[unit];
    const float  v = x[x_ptr[unit] + col];
    float       *r = y + y_ptr[unit];
    int          m = shape0[unit];
    int          n = shape1[unit];

    for (int row = 0; row < m; ++row)
        atomicAdd(&r[row], w[row * n + col] * v);

    /* bias node: last column triggers setting the bias output to 1.0 */
    if (col == n - 1)
        r[m] = 1.0f;
}

/* ── Sigmoid (standard) ───────────────────────────────────────────────────── */
__global__ void k_sigmoid_fast(
    float       * __restrict__ mem,
    const int   *ptr,
    const float *beta,
    const int   *shape0,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    float *r = mem + ptr[i];
    float  b = beta[i];
    int    m = shape0[i];
    for (int k = 0; k < m; ++k)
        r[k] = 1.0f / (1.0f + expf(-b * r[k]));
}

/* ── Rational polynomial sigmoid ─────────────────────────────────────────── */
/*   f(x) = x / (2*(|x|+1)) + 0.5                                            */
__global__ void k_sigmoid_poly_fast(
    float       * __restrict__ mem,
    const int   *ptr,
    const float *beta,
    const int   *shape0,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    float *r = mem + ptr[i];
    int    m = shape0[i];
    (void)beta;  /* beta not used for poly variant */
    for (int k = 0; k < m; ++k) {
        float v = r[k];
        r[k] = (v / (2.0f * (fabsf(v) + 1.0f))) + 0.5f;
    }
}

/* ── Sign function ───────────────────────────────────────────────────────── */
__global__ void k_sgn(float *mem, int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    mem[i] = copysignf(1.0f, mem[i]);
}

/* ── Backward: W^T * delta (build intermediate matrix) ───────────────────── */
__global__ void k_dot_transpose_fast(
    const float * __restrict__ W,
          float * __restrict__ W_buf,
    const float * __restrict__ delta,
    const int *w_ptr,
    const int *d_ptr,
    const int *shape0,
    const int *shape1,
    const int *obj_id,
    const int *col_id,
    int total_threads)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    int unit = obj_id[tid];
    int col  = col_id[tid];

    const float *w  = W     + w_ptr[unit];
          float *wb = W_buf + w_ptr[unit];
    const float *d  = delta + d_ptr[unit];
    int m0 = shape0[unit];
    int m1 = shape1[unit];

    for (int row = 0; row < m0; ++row)
        wb[row * m1 + col] = w[row * m1 + col] * d[row];
}

/* ── Column-sum of intermediate matrix -> error vector ───────────────────── */
__global__ void k_sum_dot_transpose(
    const float * __restrict__ W_buf,
          float * __restrict__ error,
    const int *w_ptr,
    const int *e_ptr,
    const int *shape0,
    const int *shape1,
    const int *obj_id,
    const int *col_id,
    int total_threads)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    int unit = obj_id[tid];
    int col  = col_id[tid];

    const float *wb = W_buf + w_ptr[unit];
    float       *e  = error + e_ptr[unit];
    int m0 = shape0[unit];
    int m1 = shape1[unit];

    float s = 0.0f;
    for (int row = 0; row < m0; ++row)
        s += wb[row * m1 + col];
    atomicAdd(&e[col], s);
}

/* ── Standard sigmoid derivative ─────────────────────────────────────────── */
/*   delta = a * (1 - a) * error                                              */
__global__ void k_sigmoid_der_mul(
    const float * __restrict__ activ,
    const float * __restrict__ error,
          float * __restrict__ delta,
    const int *a_ptr,
    const int *e_ptr,
    const int *d_ptr,
    const int *shape0,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    const float *a = activ + a_ptr[i];
    const float *e = error + e_ptr[i];
    float       *d = delta + d_ptr[i];
    int          m = shape0[i];
    for (int k = 0; k < m; ++k)
        d[k] = a[k] * (1.0f - a[k]) * e[k];
}

/* ── Rational poly sigmoid derivative ────────────────────────────────────── */
/*   For f(x) = x/(2(|x|+1)) + 0.5:  f'(x) = 1 / (2*(|x|+1)^2)              */
__global__ void k_sigmoid_poly_der_mul(
    const float * __restrict__ activ,
    const float * __restrict__ error,
          float * __restrict__ delta,
    const int *a_ptr,
    const int *e_ptr,
    const int *d_ptr,
    const int *shape0,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    const float *a = activ + a_ptr[i];
    const float *e = error + e_ptr[i];
    float       *d = delta + d_ptr[i];
    int          m = shape0[i];
    for (int k = 0; k < m; ++k) {
        /* Invert: given a = x/(2(|x|+1))+0.5, recover x */
        float t  = 2.0f * a[k] - 1.0f;
        float sg = copysignf(1.0f, t);
        float xv = t / (1.0f - sg * t);    /* x in terms of a */
        float xp = fabsf(xv) + 1.0f;
        d[k] = (1.0f / (2.0f * xp * xp)) * e[k];
    }
}

/* ── Weight update with momentum ─────────────────────────────────────────── */
/*   dW_new[l,k] = momentum * dW_prev[l,k] + lr * delta[l] * input[k]        */
__global__ void k_outer_update(
    const float * __restrict__ delta,
    const float * __restrict__ input,
    const float * __restrict__ dW_prev,
          float * __restrict__ dW,
    const int   *d_ptr,
    const int   *i_ptr,
    const int   *dW_ptr,
    const int   *shape0,
    const int   *shape1,
    const float *lr,
    const float *mom,
    const int   *obj_id,
    const int   *col_id,
    int total_threads)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    int unit = obj_id[tid];
    int col  = col_id[tid];

    const float *del   = delta   + d_ptr[unit];
    const float  inp   = input[i_ptr[unit] + col];
    const float *dp    = dW_prev + dW_ptr[unit];
    float       *dw    = dW      + dW_ptr[unit];
    float alpha  = lr [unit];
    float beta_m = mom[unit];
    int   m0     = shape0[unit];
    int   m1     = shape1[unit];

    for (int row = 0; row < m0; ++row) {
        int idx  = row * m1 + col;
        dw[idx] = fmaf(beta_m, dp[idx], alpha * del[row] * inp);
    }
}

/* ── Fused weight accumulation ───────────────────────────────────────────── */
__global__ void k_weight_add(float * __restrict__ W, const float * __restrict__ dW, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    W[i] += dW[i];
}

/* ── Data flow: copy variable-length blocks ──────────────────────────────── */
__global__ void k_copy_blocks(
    const float * __restrict__ from,
          float * __restrict__ to,
    const int *from_ptr,
    const int *from_qnt,
    const int *to_ptr,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int n = from_qnt[i];
    const float *f = from + from_ptr[i];
    float       *t = to   + to_ptr[i];
    for (int j = 0; j < n; ++j) t[j] = f[j];
}

__global__ void k_copy_blocks_comp(
    const float * __restrict__ from,
          float * __restrict__ to,
    const int *from_ptr,
    const int *from_qnt,
    const int *to_ptr,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int n = from_qnt[i];
    const float *f = from + from_ptr[i];
    float       *t = to   + to_ptr[i];
    for (int j = 0; j < n; ++j)
        t[j] = fmaf(0.8f, f[j], 0.1f); /* compress to (0.1, 0.9) */
}

__global__ void k_copy_repr_blocks(
    const float * __restrict__ from,
          float * __restrict__ to,
    const int *from_ptr,
    const int *to_ptr,
    const int *size,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    const float *f = from + from_ptr[i];
    float       *t = to   + to_ptr[i];
    for (int j = 0; j < size[i]; ++j) t[j] = f[j];
}

/* ── Frame distribution: split frame into patches ────────────────────────── */
__global__ void k_dist_frame(
    const float * __restrict__ frame,
          float * __restrict__ input_mem,
    const int *unit_ptr,
    int shape0, int shape1,   /* frame H, W */
    int dx, int dy,           /* tile grid dimensions */
    int sx, int sy,           /* tile size in pixels */
    int input_offset,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int x_block = i / dy, y_block = i % dy;
    int x0 = x_block * sx,  y0 = y_block * sy;
    float *mem = input_mem + unit_ptr[i] + input_offset;

    for (int j = 0; j < sy; ++j)
    for (int k = 0; k < sx; ++k) {
        int x = x0 + k, y = y0 + j;
        int fi  = 3 * (shape0 * x + y);
        int mi  = 3 * (k * sy  + j);
        mem[mi  ] = frame[fi  ];
        mem[mi+1] = frame[fi+1];
        mem[mi+2] = frame[fi+2];
    }
}

/* ── Collect prediction frame from output buffers ────────────────────────── */
__global__ void k_collect_frame(
          float * __restrict__ frame,
    const float * __restrict__ output_mem,
    const int *unit_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int input_offset,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int x_block = i / dy, y_block = i % dy;
    int x0 = x_block * sx,  y0 = y_block * sy;
    const float *mem = output_mem + unit_ptr[i] + input_offset;

    for (int j = 0; j < sy; ++j)
    for (int k = 0; k < sx; ++k) {
        int x = x0 + k, y = y0 + j;
        int fi  = 3 * (shape0 * x + y);
        int mi  = 3 * (k * sy  + j);
        frame[fi  ] = mem[mi  ];
        frame[fi+1] = mem[mi+1];
        frame[fi+2] = mem[mi+2];
    }
}

/* ── Collect layer activation into uint32 image ──────────────────────────── */
__global__ void k_collect_activ(
    unsigned int * __restrict__ frame,
    const float  * __restrict__ repr_mem,
    const int    *repr_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int ptr_offset,
    int total_obj)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_obj) return;
    int x_block = i / dy, y_block = i % dy;
    int x0 = x_block * sx,  y0 = y_block * sy;
    const float *mem = repr_mem + repr_ptr[i + ptr_offset];

    for (int j = 0; j < sy; ++j)
    for (int k = 0; k < sx; ++k) {
        int x = x0 + k, y = y0 + j;
        unsigned int v = (unsigned int)(255.0f * mem[k * sy + j]);
        frame[shape0 * x + y] = (v < 255u) ? v : 255u;
    }
}
