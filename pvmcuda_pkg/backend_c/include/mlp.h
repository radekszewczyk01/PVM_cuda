#pragma once

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MLP_MAX_LAYERS  10
#define MLP_MAX_ACTIV   11   /* layers + 1 */
#define MLP_MAGIC       "MLP1"

/* =========================================================================
 *  MLPCollection
 *  -------------
 *  Manages a batch of identically-structured multi-layer perceptrons
 *  (up to 10 weight layers).  Used by the Readout module.
 * ========================================================================= */

typedef struct {
    int max_layers;                 /* number of weight layers              */
    int total_objects;              /* number of MLP instances              */
    int threads;                    /* CUDA threads-per-block (sigmoid etc) */
    int grid_size;                  /* CUDA grid size (sigmoid etc)         */
    int flip_buf;                   /* toggles momentum buffer              */
    int propagate_all_the_way;      /* propagate error to input layer       */

    /* ---- per weight layer (0 .. max_layers-1) -------------------------- */
    float *h_weight_mem[MLP_MAX_LAYERS];
    float *h_dweight_mem[MLP_MAX_LAYERS];
    float *h_mweight_mem[MLP_MAX_LAYERS];
    float *h_weight_buf[MLP_MAX_LAYERS];    /* buf for dot-transpose       */
    int   *h_weight_ptr[MLP_MAX_LAYERS];
    int   *h_shape0[MLP_MAX_LAYERS];
    int   *h_shape1[MLP_MAX_LAYERS];
    float *h_beta[MLP_MAX_LAYERS];
    float *h_learning_rate[MLP_MAX_LAYERS];
    float *h_momentum[MLP_MAX_LAYERS];
    int   *h_obj_id[MLP_MAX_LAYERS];
    int   *h_row_id[MLP_MAX_LAYERS];
    int    total_threads[MLP_MAX_LAYERS];
    int    block_size[MLP_MAX_LAYERS];
    int    grid[MLP_MAX_LAYERS];

    float *d_weight_mem[MLP_MAX_LAYERS];
    float *d_dweight_mem[MLP_MAX_LAYERS];
    float *d_mweight_mem[MLP_MAX_LAYERS];
    float *d_weight_buf[MLP_MAX_LAYERS];
    int   *d_weight_ptr[MLP_MAX_LAYERS];
    int   *d_shape0[MLP_MAX_LAYERS];
    int   *d_shape1[MLP_MAX_LAYERS];
    float *d_beta[MLP_MAX_LAYERS];
    float *d_learning_rate[MLP_MAX_LAYERS];
    float *d_momentum[MLP_MAX_LAYERS];
    int   *d_obj_id[MLP_MAX_LAYERS];
    int   *d_row_id[MLP_MAX_LAYERS];

    /* ---- per activation layer (0 .. max_layers, i.e. layers+1) --------- */
    float *h_input_mem[MLP_MAX_ACTIV];
    int   *h_input_ptr[MLP_MAX_ACTIV];
    float *h_delta_mem[MLP_MAX_ACTIV];
    int   *h_delta_ptr[MLP_MAX_ACTIV];
    float *h_error_mem[MLP_MAX_ACTIV];
    int   *h_error_ptr[MLP_MAX_ACTIV];

    float *d_input_mem[MLP_MAX_ACTIV];
    int   *d_input_ptr[MLP_MAX_ACTIV];
    float *d_delta_mem[MLP_MAX_ACTIV];
    int   *d_delta_ptr[MLP_MAX_ACTIV];
    float *d_error_mem[MLP_MAX_ACTIV];
    int   *d_error_ptr[MLP_MAX_ACTIV];

    int    input_mem_size[MLP_MAX_ACTIV];

    /* ---- bookkeeping --------------------------------------------------- */
    int    layerwise_weight_mem_req[MLP_MAX_LAYERS];
    int    layerwise_weight_objects[MLP_MAX_LAYERS];
    int    layerwise_output_mem_req[MLP_MAX_LAYERS];
    int    input_layer_req;
} MLPCollection;

/* ---- public API -------------------------------------------------------- */

MLPCollection* mlp_create(int n_specs, int **specs, int *spec_lengths,
                           float learning_rate, float momentum);
void mlp_destroy(MLPCollection *mlp);

void mlp_forward(MLPCollection *mlp, int poly);
void mlp_backward(MLPCollection *mlp, int poly,
                  int propagate_all_way, int propagate_only);

void mlp_set_input(MLPCollection *mlp, const float *data);
void mlp_set_learning_rate(MLPCollection *mlp, float rate);

void mlp_generate_gpu_mem(MLPCollection *mlp);
void mlp_set_gpu_mem(MLPCollection *mlp);
void mlp_get_gpu_mem(MLPCollection *mlp);

int          mlp_save(MLPCollection *mlp, const char *filename);
MLPCollection* mlp_load(const char *filename);
