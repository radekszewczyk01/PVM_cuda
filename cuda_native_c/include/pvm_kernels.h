/* pvm_kernels.h – CUDA kernel declarations for PVM (C-compatible header)
 * Kernels are only launched from pvm_object.cu.
 * This header uses __global__ qualifiers; include only from .cu translation units.
 */
#ifndef PVM_KERNELS_H
#define PVM_KERNELS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Forward pass ─────────────────────────────────────────────────────────── */

/* Dot-product + bias: W[shape0 x shape1] * x[shape1] -> y[shape0]
 * One thread per (unit, input_col).  Uses atomicAdd into output.          */
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
    int total_threads
);

/* Standard sigmoid (in-place, per object) */
__global__ void k_sigmoid_fast(
    float       * __restrict__ mem,
    const int   *ptr,
    const float *beta,
    const int   *shape0,
    int total_obj
);

/* Rational polynomial sigmoid – faster than expf (in-place) */
__global__ void k_sigmoid_poly_fast(
    float       * __restrict__ mem,
    const int   *ptr,
    const float *beta,
    const int   *shape0,
    int total_obj
);

/* Element-wise sign function (for abs-diff error) */
__global__ void k_sgn(float *mem, int total_obj);

/* ── Backward pass ────────────────────────────────────────────────────────── */

/* Build intermediate matrix: W_buf[l,k] = W[l,k] * delta[l] */
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
    int total_threads
);

/* Column-sum W_buf -> error vector */
__global__ void k_sum_dot_transpose(
    const float * __restrict__ W_buf,
          float * __restrict__ error,
    const int *w_ptr,
    const int *e_ptr,
    const int *shape0,
    const int *shape1,
    const int *obj_id,
    const int *col_id,
    int total_threads
);

/* delta = activ * (1 - activ) * error  (standard sigmoid derivative) */
__global__ void k_sigmoid_der_mul(
    const float * __restrict__ activ,
    const float * __restrict__ error,
          float * __restrict__ delta,
    const int *a_ptr,
    const int *e_ptr,
    const int *d_ptr,
    const int *shape0,
    int total_obj
);

/* delta = poly_sigmoid_derivative(activ) * error */
__global__ void k_sigmoid_poly_der_mul(
    const float * __restrict__ activ,
    const float * __restrict__ error,
          float * __restrict__ delta,
    const int *a_ptr,
    const int *e_ptr,
    const int *d_ptr,
    const int *shape0,
    int total_obj
);

/* Weight update with momentum:
 *   dW_new = momentum * dW_prev + lr * outer(delta, input)
 *   W += dW_new                                                           */
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
    int total_threads
);

/* W += dW (fused accumulation over all weights) */
__global__ void k_weight_add(float * __restrict__ W, const float * __restrict__ dW, int n);

/* ── Data flow ────────────────────────────────────────────────────────────── */

/* Copy variable-length blocks between flat GPU arrays */
__global__ void k_copy_blocks(
    const float * __restrict__ from,
          float * __restrict__ to,
    const int *from_ptr,
    const int *from_qnt,
    const int *to_ptr,
    int total_obj
);

/* Same but compresses values to (0.1, 0.9): t[j] = 0.8*f[j] + 0.1 */
__global__ void k_copy_blocks_comp(
    const float * __restrict__ from,
          float * __restrict__ to,
    const int *from_ptr,
    const int *from_qnt,
    const int *to_ptr,
    int total_obj
);

/* Copy repr blocks (used for context in complex-layer mode) */
__global__ void k_copy_repr_blocks(
    const float * __restrict__ from,
          float * __restrict__ to,
    const int *from_ptr,
    const int *to_ptr,
    const int *size,
    int total_obj
);

/* Distribute frame patches to layer-0 unit input buffers */
__global__ void k_dist_frame(
    const float * __restrict__ frame,
          float * __restrict__ input_mem,
    const int *unit_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int input_offset,
    int total_obj
);

/* Collect prediction frame from output buffers */
__global__ void k_collect_frame(
          float * __restrict__ frame,
    const float * __restrict__ output_mem,
    const int *unit_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int input_offset,
    int total_obj
);

/* Collect layer activation into uint32 image (for visualization) */
__global__ void k_collect_activ(
    unsigned int * __restrict__ frame,
    const float  * __restrict__ repr_mem,
    const int    *repr_ptr,
    int shape0, int shape1,
    int dx, int dy,
    int sx, int sy,
    int ptr_offset,
    int total_obj
);

#ifdef __cplusplus
}
#endif

#endif /* PVM_KERNELS_H */
