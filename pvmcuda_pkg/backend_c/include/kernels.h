#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

/* Matric-vector products */

__global__ void gpu_dot(float *mem1, float *mem2, float *mem3,
                        int *ptr1, int *ptr2, int *ptr3,
                        int *shape0, int *shape1, int total_obj);

__global__ void gpu_dot_transpose(float *mem1, float *mem2, float *mem3,
                                  int *ptr1, int *ptr2, int *ptr3,
                                  int *shape0, int *shape1, int total_obj);

__global__ void gpu_dot_transpose_fast(float *mem0, float *mem1, float *mem2,
                                       int *ptr1, int *ptr2,
                                       int *shape0, int *shape1,
                                       int *obj_id, int *row_id,
                                       int total_threads);

__global__ void gpu_sum_dot_transpose(float *mem0, float *mem1,
                                      int *ptr1, int *ptr3,
                                      int *shape0, int *shape1,
                                      int *obj_id, int *row_id,
                                      int total_threads);

__global__ void gpu_dot_sigmoid(float *mem1, float *mem2, float *mem3,
                                int *ptr1, int *ptr2, int *ptr3,
                                float *beta, int *shape0, int *shape1,
                                int total_obj);

__global__ void gpu_dot_fast(float *mem1, float *mem2, float *mem3,
                             int *ptr1, int *ptr2, int *ptr3,
                             int *shape0, int *shape1,
                             int *obj_id, int *row_id,
                             int total_threads);

__global__ void gpu_dot_fast_set_bias(float *mem1, float *mem2, float *mem3,
                                      int *ptr1, int *ptr2, int *ptr3,
                                      int *shape0, int *shape1,
                                      int *obj_id, int *row_id,
                                      int total_threads);

__global__ void gpu_dot_slow(float *mem1, float *mem2, float *mem3,
                             int *ptr1, int *ptr2, int *ptr3,
                             float *beta, int *shape0, int *shape1,
                             int total_obj);

__global__ void gpu_dot_sigmoid_poly(float *mem1, float *mem2, float *mem3,
                                     int *ptr1, int *ptr2, int *ptr3,
                                     int *shape0, int *shape1, int total_obj);

/* Activation functions */

__global__ void gpu_sgn(float *mem1, int total_obj);

__global__ void gpu_sigmoid_fast(float *mem1, int *ptr1, float *beta,
                                 int *shape0, int total_obj);

__global__ void gpu_sigmoid_poly_fast(float *mem1, int *ptr1, float *beta,
                                      int *shape0, int total_obj);

/* Activation derivatives */

__global__ void gpu_sigmoid_der_mul(float *mem1, float *mem2, float *mem3,
                                    int *ptr1, int *ptr2, int *ptr3,
                                    int *shape0, int total_obj);

__global__ void gpu_sigmoid_poly_der_mul(float *mem1, float *mem2, float *mem3,
                                         int *ptr1, int *ptr2, int *ptr3,
                                         int *shape0, int total_obj);

/* Other products */

__global__ void gpu_outer_simple(float *mem1, float *mem2, float *mem3,
                                 int *ptr1, int *ptr2, int *ptr3,
                                 int *shape0, int *shape1, int total_obj);

__global__ void gpu_generalized_outer(float *mem1, float *mem2,
                                      float *mem3, float *mem4,
                                      int *ptr1, int *ptr2,
                                      int *ptr3, int *ptr4,
                                      int *shape0, int *shape1,
                                      float *alpha, float *beta,
                                      int total_obj);

__global__ void gpu_generalized_outer_fast(float *mem1, float *mem2,
                                           float *mem3, float *mem4,
                                           int *ptr1, int *ptr2,
                                           int *ptr3, int *ptr4,
                                           int *shape0, int *shape1,
                                           float *alpha, float *beta,
                                           int *obj_id, int *row_id,
                                           int total_threads);

__global__ void gpu_generalized_outer_fast2(float *mem1, float *mem2,
                                            float *mem3, float *mem4,
                                            int *ptr1, int *ptr2,
                                            int *ptr3, int *ptr4,
                                            int *shape0, int *shape1,
                                            float alpha, float beta,
                                            int *obj_id, int *row_id,
                                            int total_threads);

__global__ void gpu_generalized_outer_fast3(float *mem1, float *mem2,
                                            float *mem3, float *mem4,
                                            int *ptr1, int *ptr2,
                                            int *ptr3, int *ptr4,
                                            int *shape0, int *shape1,
                                            float *alpha, float *beta,
                                            int *obj_id, int *row_id,
                                            int total_threads);

/* Element-wise arithmetic */

__global__ void gpu_add(float *mem1, float *mem2,
                        int *ptr1, int *ptr2,
                        int *shape0, int *shape1, int total_obj);

__global__ void gpu_mov(float *mem1, float *mem2,
                        int *ptr1, int *ptr2,
                        int *shape0, int *shape1, int total_obj);

__global__ void gpu_clip(float *from_arr, float val0, float val1,
                         int total_obj);

/* Frame distribute / collect */

__global__ void gpu_dist_frame(float *frame_arr, float *input_obj_mem,
                               int *ptr2, int shape0, int shape1,
                               int dx, int dy, int sx, int sy,
                               int input_offset, int total_obj);

__global__ void gpu_dist_frame4(float *frame_arr, float *input_obj_mem,
                                int *ptr2, int shape0, int shape1,
                                int dx, int dy, int sx, int sy,
                                int input_offset, int total_obj);

__global__ void gpu_calc_error_frame(float *frame_arr, float *out_obj_mem,
                                     int *ptr2, float *error_obj_mem,
                                     int *ptr3, int shape0, int shape1,
                                     int dx, int dy, int sx, int sy,
                                     int input_offset, int total_obj);

__global__ void gpu_calc_error_frame_1ch(float *frame_arr, float *out_obj_mem,
                                         int *ptr2, float *error_obj_mem,
                                         int *ptr3, int shape0, int shape1,
                                         int dx, int dy, int sx, int sy,
                                         int input_offset, int total_obj);

__global__ void gpu_calc_abs_diff_error_frame_1ch(float *frame_arr,
                                                   float *out_obj_mem,
                                                   int *ptr2,
                                                   float *error_obj_mem,
                                                   int *ptr3,
                                                   int shape0, int shape1,
                                                   int dx, int dy,
                                                   int sx, int sy,
                                                   int input_offset,
                                                   int total_obj);

__global__ void gpu_collect_frame4(float *frame_arr, float *input_obj_mem,
                                   int *ptr2, int shape0, int shape1,
                                   int dx, int dy, int sx, int sy,
                                   int input_offset, int total_obj);

__global__ void gpu_collect_frame(float *frame_arr, float *input_obj_mem,
                                  int *ptr2, int shape0, int shape1,
                                  int dx, int dy, int sx, int sy,
                                  int input_offset, int total_obj);

__global__ void gpu_collect_frame_1ch(float *frame_arr, float *input_obj_mem,
                                      int *ptr2, int shape0, int shape1,
                                      int dx, int dy, int sx, int sy,
                                      int input_offset, int total_obj);

__global__ void gpu_collect_activ(unsigned int *frame_arr,
                                  float *input_obj_mem, int *ptr2,
                                  int shape0, int shape1,
                                  int dx, int dy, int sx, int sy,
                                  int ptr_offset, int total_obj);

/* Block copy utilities */

__global__ void gpu_copy_blocks(float *from_arr, float *to_arr,
                                int *from_ptr, int *from_qnt, int *to_ptr,
                                int total_obj);

__global__ void gpu_copy_blocks_fixed(float *from_arr, float *to_arr,
                                      int *from_ptr, int *to_ptr,
                                      int from_qnt, float mul,
                                      int total_obj);

__global__ void gpu_set_one_hot_error(float *from_arr, int *from_ptr,
                                      float *to_arr, int *to_ptr,
                                      int hot, int length, int total_obj);

__global__ void gpu_copy_blocks_comp(float *from_arr, float *to_arr,
                                     int *from_ptr, int *from_qnt,
                                     int *to_ptr, int total_obj);

__global__ void gpu_copy_blocks_sigmoid(float *from_arr, float *to_arr,
                                        int *from_ptr, int *from_qnt,
                                        int *to_ptr, float beta,
                                        int total_obj);

/* Host utilities */

void gpu_array_copy(float *dst, float *src, size_t nbytes);
void gpu_memset_zero(float *ptr, size_t nbytes);
