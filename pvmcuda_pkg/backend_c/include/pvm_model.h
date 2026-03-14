#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define PVM_MAX_LAYERS 16
#define PVM_SEQ_LENGTH 3
#define PVM_SEQ_INTERVAL 2

typedef struct {
    int num_layers;
    int layer_shapes[PVM_MAX_LAYERS];
    int input_block_size;
    int hidden_block_size;
    int input_channels;
    int lateral_radius;
    int context_exclude_self;
    int fan_in_square_size;
    int fan_in_radius;
    int feed_context_in_complex_layer;
    int send_context_two_layers_back;
    int last_layer_context_to_all;
    int polynomial;
    float initial_learning_rate;
    float final_learning_rate;
    float intermediate_learning_rate;
    int delay_each_layer_learning;
    int delay_final_learning_rate;
    int delay_intermediate_learning_rate;
    float momentum_val;
    int opt_abs_diff;
    int ignore_depth;
} PVMConfig;

typedef struct {
    int id;
    int layer;
    int grid_x, grid_y;
    int base_input_offset;
    int base_input_size;
    int output_dim;
    int input_dim;
    int context_dim;
    int size;        // hidden_block_size^2
    int base_context_offset;   // = size + 1
    int running_input_ptr;      // used during flow ptr generation

    int n_primary_sources;
    int n_context_sources;
    int n_primary_destinations;
    int n_context_destinations;
    int *primary_sources;       // dynamic array of block IDs
    int *context_sources;       // dynamic array of block IDs
    int *primary_destinations;  // dynamic array of block IDs
    int *context_destinations;  // dynamic array of block IDs

    int *xs;   // unique x values (dynamic array)
    int *ys;   // unique y values (dynamic array)
    int n_xs, n_ys;

    int w0_ptr;   // weight pointer for layer 0
    int w1_ptr;   // weight pointer for layer 1
    int i_ptr;    // input pointer
    int r_ptr;    // representation pointer
} PVMBlock;

typedef struct {
    PVMConfig config;
    char name[256];
    char uniq_id[16];
    char time_stamp[64];
    char device_name[256];
    int step;
    int poly;
    int buffer_index;

    // Topology
    PVMBlock *graph;
    int *layer_ptrs;
    int total_units;
    int total_weights;
    int total_input_mem;
    int total_repr_mem;
    int total_primary_projections;
    int total_context_projections;
    float learning_rate;
    float momentum;
    int input_channels;

    // Host arrays (for save/load and staging)
    float *h_weight_mem;
    float *h_dweight_mem[2];
    float *h_weight_cache;
    float *h_input_activation[PVM_SEQ_LENGTH];
    float *h_input_delta[PVM_SEQ_LENGTH];
    float *h_input_error[PVM_SEQ_LENGTH];
    float *h_output_activation[PVM_SEQ_LENGTH];
    float *h_output_delta[PVM_SEQ_LENGTH];
    float *h_output_error[PVM_SEQ_LENGTH];
    float *h_repr_activation[PVM_SEQ_LENGTH];
    float *h_repr_delta[PVM_SEQ_LENGTH];
    float *h_repr_error[PVM_SEQ_LENGTH];
    float *h_beta_input;
    float *h_beta_repr;
    float *h_learning_rate_arr;
    float *h_momentum_arr;

    // Host pointer/shape arrays
    int *h_weight_ptr0;
    int *h_weight_ptr1;
    int *h_input_ptr;
    int *h_repr_ptr;
    int *h_shape0_L0;
    int *h_shape1_L0;
    int *h_shape0_L1;
    int *h_shape1_L1;
    int *h_obj_id_L0;
    int *h_row_id_L0;
    int *h_obj_id_L1;
    int *h_row_id_L1;

    // Host flow pointers
    int *h_flow_from;
    int *h_flow_to;
    int *h_flow_size;
    int *h_flow_repr_from;
    int *h_flow_repr_to;
    int *h_flow_repr_size;
    int *h_flow_shift_from;
    int *h_flow_shift_to;
    int *h_flow_shift_size;
    int *h_flow_input_frame;
    int *h_flow_input_frame_size;

    // Device arrays
    float *d_weight_mem;
    float *d_dweight_mem[2];
    float *d_weight_cache;
    float *d_input_activation[PVM_SEQ_LENGTH];
    float *d_input_delta[PVM_SEQ_LENGTH];
    float *d_input_error[PVM_SEQ_LENGTH];
    float *d_output_activation[PVM_SEQ_LENGTH];
    float *d_output_delta[PVM_SEQ_LENGTH];
    float *d_output_error[PVM_SEQ_LENGTH];
    float *d_repr_activation[PVM_SEQ_LENGTH];
    float *d_repr_delta[PVM_SEQ_LENGTH];
    float *d_repr_error[PVM_SEQ_LENGTH];
    float *d_beta_input;
    float *d_beta_repr;
    float *d_learning_rate_arr;
    float *d_momentum_arr;

    // Device pointer/shape arrays
    int *d_weight_ptr0;
    int *d_weight_ptr1;
    int *d_input_ptr;
    int *d_repr_ptr;
    int *d_shape0_L0;
    int *d_shape1_L0;
    int *d_shape0_L1;
    int *d_shape1_L1;
    int *d_obj_id_L0;
    int *d_row_id_L0;
    int *d_obj_id_L1;
    int *d_row_id_L1;

    // Device flow pointers
    int *d_flow_from;
    int *d_flow_to;
    int *d_flow_size;
    int *d_flow_repr_from;
    int *d_flow_repr_to;
    int *d_flow_repr_size;
    int *d_flow_shift_from;
    int *d_flow_shift_to;
    int *d_flow_shift_size;
    int *d_flow_input_frame;
    int *d_flow_input_frame_size;

    // CUDA launch params
    int cuda_block_size;
    int cuda_grid_L0;
    int cuda_grid_L1;
    int cuda_total_threads_k2_L0;
    int cuda_total_threads_k2_L1;

    // For freeze/unfreeze
    float *h_learning_rate_saved;
    float *h_momentum_saved;
} PVMObject;

PVMObject* pvm_object_create(const PVMConfig *config, const char *name);
void pvm_object_destroy(PVMObject *pvm);
void pvm_generate_graph(PVMObject *pvm);
void pvm_generate_memory(PVMObject *pvm);
void pvm_generate_memory_ptrs(PVMObject *pvm);
void pvm_generate_flow_ptrs(PVMObject *pvm);
void pvm_create_gpu_mem(PVMObject *pvm);
void pvm_update_learning_rate(PVMObject *pvm, float override_rate);
void pvm_push_input(PVMObject *pvm, const float *frame, int h, int w, int ch);
void pvm_forward(PVMObject *pvm);
void pvm_backward(PVMObject *pvm);
void pvm_pop_prediction(PVMObject *pvm, float *out_buf, int delta_step);
void pvm_pop_layer(PVMObject *pvm, unsigned char *out_buf, int layer);
void pvm_get_input_shape(PVMObject *pvm, int *w, int *h, int *ch);
void pvm_freeze_learning(PVMObject *pvm);
void pvm_unfreeze_learning(PVMObject *pvm);
void pvm_get_data_from_gpu(PVMObject *pvm);
int pvm_save(PVMObject *pvm, const char *filename);
PVMObject* pvm_load(const char *filename);
