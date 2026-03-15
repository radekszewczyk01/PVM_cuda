#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

/* Matric-vector products */

__global__ void gpu_dot(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                        int * __restrict__ ptr1, int * __restrict__ ptr2, int * __restrict__ ptr3,
                        int * __restrict__ shape0, int * __restrict__ shape1, int total_obj);

__global__ void gpu_dot_transpose(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                                  int * __restrict__ ptr1, int * __restrict__ ptr2, int * __restrict__ ptr3,
                                  int * __restrict__ shape0, int * __restrict__ shape1, int total_obj);

__global__ void gpu_dot_transpose_fast(float * __restrict__ mem0, float * __restrict__ mem1, float * __restrict__ mem2,
                                       int * __restrict__ ptr1, int * __restrict__ ptr2,
                                       int * __restrict__ shape0, int * __restrict__ shape1,
                                       int * __restrict__ obj_id, int * __restrict__ row_id,
                                       int total_threads);

__global__ void gpu_sum_dot_transpose(float * __restrict__ mem0, float * __restrict__ mem1,
                                      int * __restrict__ ptr1, int * __restrict__ ptr3,
                                      int * __restrict__ shape0, int * __restrict__ shape1,
                                      int * __restrict__ obj_id, int * __restrict__ row_id,
                                      int total_threads);

__global__ void gpu_dot_sigmoid(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                                int * __restrict__ ptr1, int * __restrict__ ptr2, int * __restrict__ ptr3,
                                float * __restrict__ beta, int * __restrict__ shape0, int * __restrict__ shape1,
                                int total_obj);

__global__ void gpu_dot_fast(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                             int * __restrict__ ptr1, int * __restrict__ ptr2, int * __restrict__ ptr3,
                             int * __restrict__ shape0, int * __restrict__ shape1,
                             int * __restrict__ obj_id, int * __restrict__ row_id,
                             int total_threads);

__global__ void gpu_dot_fast_set_bias(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                                      int * __restrict__ ptr1, int * __restrict__ ptr2, int * __restrict__ ptr3,
                                      int * __restrict__ shape0, int * __restrict__ shape1,
                                      int * __restrict__ obj_id, int * __restrict__ row_id,
                                      int total_threads);

__global__ void gpu_dot_slow(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                             int * __restrict__ ptr1, int * __restrict__ ptr2, int * __restrict__ ptr3,
                             float * __restrict__ beta, int * __restrict__ shape0, int * __restrict__ shape1,
                             int total_obj);

__global__ void gpu_dot_sigmoid_poly(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                                     int * __restrict__ ptr1, int * __restrict__ ptr2, int * __restrict__ ptr3,
                                     int * __restrict__ shape0, int * __restrict__ shape1, int total_obj);

/* Activation functions */

__global__ void gpu_sgn(float * __restrict__ mem1, int total_obj);

__global__ void gpu_sigmoid_fast(float * __restrict__ mem1, int * __restrict__ ptr1, float * __restrict__ beta,
                                 int * __restrict__ shape0, int total_obj);

__global__ void gpu_sigmoid_poly_fast(float * __restrict__ mem1, int * __restrict__ ptr1, float * __restrict__ beta,
                                      int * __restrict__ shape0, int total_obj);

/* Activation derivatives */

__global__ void gpu_sigmoid_der_mul(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                                    int *ptr1, int *ptr2, int *ptr3,
                                    int * __restrict__ shape0, int total_obj);

__global__ void gpu_sigmoid_poly_der_mul(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                                         int *ptr1, int *ptr2, int *ptr3,
                                         int * __restrict__ shape0, int total_obj);

/* Other products */

__global__ void gpu_outer_simple(float * __restrict__ mem1, float * __restrict__ mem2, float * __restrict__ mem3,
                                 int * __restrict__ ptr1, int * __restrict__ ptr2, int * __restrict__ ptr3,
                                 int * __restrict__ shape0, int * __restrict__ shape1, int total_obj);

__global__ void gpu_generalized_outer(float * __restrict__ mem1, float * __restrict__ mem2,
                                      float * __restrict__ mem3, float * __restrict__ mem4,
                                      int * __restrict__ ptr1, int * __restrict__ ptr2,
                                      int * __restrict__ ptr3, int * __restrict__ ptr4,
                                      int * __restrict__ shape0, int * __restrict__ shape1,
                                      float * __restrict__ alpha, float * __restrict__ beta,
                                      int total_obj);

__global__ void gpu_generalized_outer_fast(float * __restrict__ mem1, float * __restrict__ mem2,
                                           float * __restrict__ mem3, float * __restrict__ mem4,
                                           int * __restrict__ ptr1, int * __restrict__ ptr2,
                                           int *ptr3, int *ptr4,
                                           int * __restrict__ shape0, int * __restrict__ shape1,
                                           float * __restrict__ alpha, float * __restrict__ beta,
                                           int * __restrict__ obj_id, int * __restrict__ row_id,
                                           int total_threads);

__global__ void gpu_generalized_outer_fast2(float * __restrict__ mem1, float * __restrict__ mem2,
                                            float * __restrict__ mem3, float * __restrict__ mem4,
                                            int * __restrict__ ptr1, int * __restrict__ ptr2,
                                            int * __restrict__ ptr3, int * __restrict__ ptr4,
                                            int * __restrict__ shape0, int * __restrict__ shape1,
                                            float alpha, float beta,
                                            int * __restrict__ obj_id, int * __restrict__ row_id,
                                            int total_threads);

__global__ void gpu_generalized_outer_fast3(float * __restrict__ mem1, float * __restrict__ mem2,
                                            float * __restrict__ mem3, float * __restrict__ mem4,
                                            int * __restrict__ ptr1, int * __restrict__ ptr2,
                                            int *ptr3, int *ptr4,
                                            int * __restrict__ shape0, int * __restrict__ shape1,
                                            float * __restrict__ alpha, float * __restrict__ beta,
                                            int * __restrict__ obj_id, int * __restrict__ row_id,
                                            int total_threads);

/* Element-wise arithmetic */

__global__ void gpu_add(float * __restrict__ mem1, float * __restrict__ mem2,
                        int *ptr1, int *ptr2,
                        int * __restrict__ shape0, int * __restrict__ shape1, int total_obj);

__global__ void gpu_mov(float * __restrict__ mem1, float * __restrict__ mem2,
                        int * __restrict__ ptr1, int * __restrict__ ptr2,
                        int * __restrict__ shape0, int * __restrict__ shape1, int total_obj);

__global__ void gpu_clip(float * __restrict__ from_arr, float val0, float val1,
                         int total_obj);

/* Frame distribute / collect */

__global__ void gpu_dist_frame(float * __restrict__ frame_arr, float * __restrict__ input_obj_mem,
                               int * __restrict__ ptr2, int shape0, int shape1,
                               int dx, int dy, int sx, int sy,
                               int input_offset, int total_obj);

__global__ void gpu_dist_frame4(float * __restrict__ frame_arr, float * __restrict__ input_obj_mem,
                                int * __restrict__ ptr2, int shape0, int shape1,
                                int dx, int dy, int sx, int sy,
                                int input_offset, int total_obj);

__global__ void gpu_calc_error_frame(float * __restrict__ frame_arr, float * __restrict__ out_obj_mem,
                                     int * __restrict__ ptr2, float * __restrict__ error_obj_mem,
                                     int * __restrict__ ptr3, int shape0, int shape1,
                                     int dx, int dy, int sx, int sy,
                                     int input_offset, int total_obj);

__global__ void gpu_calc_error_frame_1ch(float * __restrict__ frame_arr, float * __restrict__ out_obj_mem,
                                         int * __restrict__ ptr2, float * __restrict__ error_obj_mem,
                                         int * __restrict__ ptr3, int shape0, int shape1,
                                         int dx, int dy, int sx, int sy,
                                         int input_offset, int total_obj);

__global__ void gpu_calc_abs_diff_error_frame_1ch(float * __restrict__ frame_arr,
                                                   float * __restrict__ out_obj_mem,
                                                   int * __restrict__ ptr2,
                                                   float * __restrict__ error_obj_mem,
                                                   int * __restrict__ ptr3,
                                                   int shape0, int shape1,
                                                   int dx, int dy,
                                                   int sx, int sy,
                                                   int input_offset,
                                                   int total_obj);

__global__ void gpu_collect_frame4(float * __restrict__ frame_arr, float * __restrict__ input_obj_mem,
                                   int * __restrict__ ptr2, int shape0, int shape1,
                                   int dx, int dy, int sx, int sy,
                                   int input_offset, int total_obj);

__global__ void gpu_collect_frame(float * __restrict__ frame_arr, float * __restrict__ input_obj_mem,
                                  int * __restrict__ ptr2, int shape0, int shape1,
                                  int dx, int dy, int sx, int sy,
                                  int input_offset, int total_obj);

__global__ void gpu_collect_frame_1ch(float * __restrict__ frame_arr, float * __restrict__ input_obj_mem,
                                      int * __restrict__ ptr2, int shape0, int shape1,
                                      int dx, int dy, int sx, int sy,
                                      int input_offset, int total_obj);

__global__ void gpu_collect_activ(unsigned int * __restrict__ frame_arr,
                                  float * __restrict__ input_obj_mem, int * __restrict__ ptr2,
                                  int shape0, int shape1,
                                  int dx, int dy, int sx, int sy,
                                  int ptr_offset, int total_obj);

/* Block copy utilities */

__global__ void gpu_copy_blocks(float * __restrict__ from_arr, float * __restrict__ to_arr,
                                int * __restrict__ from_ptr, int * __restrict__ from_qnt, int * __restrict__ to_ptr,
                                int total_obj);

__global__ void gpu_copy_blocks_fixed(float * __restrict__ from_arr, float * __restrict__ to_arr,
                                      int * __restrict__ from_ptr, int * __restrict__ to_ptr,
                                      int from_qnt, float mul,
                                      int total_obj);

__global__ void gpu_set_one_hot_error(float * __restrict__ from_arr, int * __restrict__ from_ptr,
                                      float * __restrict__ to_arr, int * __restrict__ to_ptr,
                                      int hot, int length, int total_obj);

__global__ void gpu_copy_blocks_comp(float * __restrict__ from_arr, float * __restrict__ to_arr,
                                     int * __restrict__ from_ptr, int * __restrict__ from_qnt,
                                     int * __restrict__ to_ptr, int total_obj);

__global__ void gpu_copy_blocks_sigmoid(float * __restrict__ from_arr, float * __restrict__ to_arr,
                                        int * __restrict__ from_ptr, int * __restrict__ from_qnt,
                                        int * __restrict__ to_ptr, float beta,
                                        int total_obj);

/* Host utilities */

void gpu_array_copy(float *dst, float *src, size_t nbytes);
void gpu_memset_zero(float *ptr, size_t nbytes);
