/*
 * PVM Model -- C/CUDA implementation of the PVM sequence learner.
 * Port of Python class PVM_object from sequence_learner.py.
 *
 * (C) 2017 Filip Piekniewski -- All Rights Reserved
 * C port 2026
 */

#include "pvm_model.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* =========================================================================
 *  Small helper kernels (file-local)
 * ========================================================================= */

__global__ void axpy_neg(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] - b[i];
}

__global__ void axpy_add(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] += a[i];
}

/* =========================================================================
 *  Dynamic-array helpers
 * ========================================================================= */

static void append_int(int **arr, int *count, int *cap, int val) {
    if (*count >= *cap) {
        /* Ensure we never shrink: new cap is at least (count+1)*2, minimum 8 */
        int new_cap = (*count + 1) * 2;
        if (new_cap < 8) new_cap = 8;
        *cap = new_cap;
        *arr = (int *)realloc(*arr, sizeof(int) * new_cap);
    }
    (*arr)[(*count)++] = val;
}

static void append_unique_int(int **arr, int *count, int *cap, int val) {
    for (int i = 0; i < *count; i++) {
        if ((*arr)[i] == val) return;
    }
    append_int(arr, count, cap, val);
}

/* =========================================================================
 *  Geometry helpers
 * ========================================================================= */

static int get_surround_c(int x, int y, int x_size, int y_size,
                           int radius, int exclude_self,
                           int *out_x, int *out_y) {
    int count = 0;
    int r = radius;
    for (int dx = -r; dx <= r; dx++) {
        for (int dy = -r; dy <= r; dy++) {
            if (dx * dx + dy * dy > r * r) continue;
            if (exclude_self && dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < x_size && ny >= 0 && ny < y_size) {
                if (out_x) out_x[count] = nx;
                if (out_y) out_y[count] = ny;
                count++;
            }
        }
    }
    return count;
}

static int get_fan_in_c(int x, int y,
                         int dim_x_l, int dim_y_l,
                         int dim_x_u, int dim_y_u,
                         int block_x, int block_y,
                         float radius,
                         int *out_x, int *out_y) {
    float factor_x, factor_y;
    if (dim_x_u > 1)
        factor_x = ((float)(dim_x_l - 1) - (float)(block_x - 1)) / (float)(dim_x_u - 1);
    else
        factor_x = ((float)(dim_x_l - 1) - (float)(block_x)) / 2.0f;

    if (dim_y_u > 1)
        factor_y = ((float)(dim_y_l - 1) - (float)(block_y - 1)) / (float)(dim_y_u - 1);
    else
        factor_y = ((float)(dim_y_l - 1) - (float)(block_y)) / 2.0f;

    int count = 0;
    float r2 = radius * radius;

    if (dim_x_u > 1 && dim_y_u > 1) {
        for (int xx = 0; xx < block_x; xx++) {
            for (int yy = 0; yy < block_y; yy++) {
                float cx = xx - (block_x - 1) * 0.5f;
                float cy = yy - (block_y - 1) * 0.5f;
                if (cx * cx + cy * cy > r2) continue;
                if (out_x) out_x[count] = (int)(factor_x * x + xx);
                if (out_y) out_y[count] = (int)(factor_y * y + yy);
                count++;
            }
        }
    } else if (dim_x_u == 1 && dim_y_u > 1) {
        for (int xx = 0; xx < block_x; xx++) {
            for (int yy = 0; yy < block_y; yy++) {
                float cx = xx - (block_x - 1) * 0.5f;
                float cy = yy - (block_y - 1) * 0.5f;
                if (cx * cx + cy * cy > r2) continue;
                if (out_x) out_x[count] = (int)((dim_x_l - block_x) / 2.0f + xx);
                if (out_y) out_y[count] = (int)(factor_y * y + yy);
                count++;
            }
        }
    } else if (dim_x_u > 1 && dim_y_u == 1) {
        for (int xx = 0; xx < block_x; xx++) {
            for (int yy = 0; yy < block_y; yy++) {
                float cx = xx - (block_x - 1) * 0.5f;
                float cy = yy - (block_y - 1) * 0.5f;
                if (cx * cx + cy * cy > r2) continue;
                if (out_x) out_x[count] = (int)(factor_x * x + xx);
                if (out_y) out_y[count] = (int)((dim_y_l - block_y) / 2.0f + yy);
                count++;
            }
        }
    } else { /* dim_x_u == 1 && dim_y_u == 1 */
        for (int xx = 0; xx < block_x; xx++) {
            for (int yy = 0; yy < block_y; yy++) {
                float cx = xx - (block_x - 1) * 0.5f;
                float cy = yy - (block_y - 1) * 0.5f;
                if (cx * cx + cy * cy > r2) continue;
                if (out_x) out_x[count] = (int)((dim_x_l - block_x) / 2.0f + xx);
                if (out_y) out_y[count] = (int)((dim_y_l - block_y) / 2.0f + yy);
                count++;
            }
        }
    }
    return count;
}

/* helper: check whether a block id already appears in a dynamic array */
static int id_in_array(const int *arr, int n, int val) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == val) return 1;
    }
    return 0;
}

/* =========================================================================
 *  Graph generation
 * ========================================================================= */

void pvm_generate_graph(PVMObject *pvm) {
    PVMConfig *c = &pvm->config;
    int seq_interval = PVM_SEQ_INTERVAL;
    int ibs = c->input_block_size;
    int hbs = c->hidden_block_size;
    int ch  = pvm->input_channels;
    int num_layers = c->num_layers;
    int *ls = c->layer_shapes;

    pvm->learning_rate = 0.001f;
    pvm->momentum      = 0.9f;

    /* Count total blocks */
    int total_units = 0;
    for (int l = 0; l < num_layers; l++)
        total_units += ls[l] * ls[l];
    pvm->total_units = total_units;

    /* Allocate graph and layer_ptrs */
    pvm->graph      = (PVMBlock *)calloc(total_units, sizeof(PVMBlock));
    pvm->layer_ptrs = (int *)calloc(num_layers, sizeof(int));

    int idx = 0;
    for (int layer = 0; layer < num_layers; layer++) {
        if (layer == 0)
            pvm->layer_ptrs[0] = 0;
        else
            pvm->layer_ptrs[layer] = pvm->layer_ptrs[layer - 1] + ls[layer - 1] * ls[layer - 1];

        for (int x = 0; x < ls[layer]; x++) {
            for (int y = 0; y < ls[layer]; y++) {
                PVMBlock *b = &pvm->graph[idx];
                memset(b, 0, sizeof(PVMBlock));
                b->id    = idx;
                b->layer = layer;
                b->grid_x = x;
                b->grid_y = y;
                b->size  = hbs * hbs;

                if (layer == 0) {
                    b->base_input_offset = seq_interval * ibs * ibs * ch;
                    b->output_dim        = b->base_input_offset;
                    b->base_input_size   = ibs * ibs * ch;
                    /* xs = [x], ys = [y] */
                    b->n_xs = 0; b->xs = NULL;
                    int xs_cap = 0;
                    append_int(&b->xs, &b->n_xs, &xs_cap, x);
                    b->n_ys = 0; b->ys = NULL;
                    int ys_cap = 0;
                    append_int(&b->ys, &b->n_ys, &ys_cap, y);
                } else {
                    b->base_input_offset = 0;
                    b->base_input_size   = 0;
                    b->output_dim        = 0;
                    b->n_xs = 0; b->xs = NULL;
                    b->n_ys = 0; b->ys = NULL;
                }

                b->input_dim   = 0;
                b->context_dim = 0;
                b->base_context_offset = b->size + 1;
                b->running_input_ptr   = b->base_input_offset;

                b->n_primary_sources      = 0; b->primary_sources      = NULL;
                b->n_context_sources      = 0; b->context_sources      = NULL;
                b->n_primary_destinations = 0; b->primary_destinations = NULL;
                b->n_context_destinations = 0; b->context_destinations = NULL;

                idx++;
            }
        }
    }

    /* Scratch buffers for surround / fan-in results */
    int max_surround = (2 * c->lateral_radius + 1) * (2 * c->lateral_radius + 1);
    int *sur_x = (int *)malloc(sizeof(int) * max_surround);
    int *sur_y = (int *)malloc(sizeof(int) * max_surround);
    int max_fan = c->fan_in_square_size * c->fan_in_square_size;
    int *fan_x = (int *)malloc(sizeof(int) * max_fan);
    int *fan_y = (int *)malloc(sizeof(int) * max_fan);

    int total_primary_projections = 0;
    int total_context_projections = 0;

    /* --- First connectivity pass: surround + fan-in --- */
    for (int bi = 0; bi < total_units; bi++) {
        PVMBlock *b = &pvm->graph[bi];
        int layer = b->layer;

        /* Lateral surround context */
        int n_sur = get_surround_c(b->grid_x, b->grid_y,
                                    ls[layer], ls[layer],
                                    c->lateral_radius,
                                    c->context_exclude_self,
                                    sur_x, sur_y);
        for (int si = 0; si < n_sur; si++) {
            int sx = sur_x[si];
            int sy = sur_y[si];
            /* Find block in same layer at (sx, sy) */
            int sub_id = pvm->layer_ptrs[layer] + sx * ls[layer] + sy;
            PVMBlock *sub = &pvm->graph[sub_id];
            int pd_cap = 0;
            append_int(&sub->context_destinations, &sub->n_context_destinations, &pd_cap, b->id);
            int cs_cap = 0;
            append_int(&b->context_sources, &b->n_context_sources, &cs_cap, sub_id);
            total_context_projections++;
        }

        /* Fan-in from layer below */
        if (layer > 0) {
            int n_fan = get_fan_in_c(b->grid_x, b->grid_y,
                                      ls[layer - 1], ls[layer - 1],
                                      ls[layer], ls[layer],
                                      c->fan_in_square_size, c->fan_in_square_size,
                                      (float)c->fan_in_radius,
                                      fan_x, fan_y);
            for (int fi = 0; fi < n_fan; fi++) {
                int fx = fan_x[fi];
                int fy = fan_y[fi];
                int sub_id = pvm->layer_ptrs[layer - 1] + fx * ls[layer - 1] + fy;
                PVMBlock *sub = &pvm->graph[sub_id];
                int pd_cap = 0;
                append_int(&sub->primary_destinations, &sub->n_primary_destinations, &pd_cap, b->id);
                int ps_cap = 0;
                append_int(&b->primary_sources, &b->n_primary_sources, &ps_cap, sub_id);
                b->base_input_size += sub->size;
                /* extend unique xs, ys */
                int xs_cap2 = 0, ys_cap2 = 0;
                for (int xi = 0; xi < sub->n_xs; xi++)
                    append_unique_int(&b->xs, &b->n_xs, &xs_cap2, sub->xs[xi]);
                for (int yi = 0; yi < sub->n_ys; yi++)
                    append_unique_int(&b->ys, &b->n_ys, &ys_cap2, sub->ys[yi]);
                total_primary_projections++;
            }
        }
    }

    free(sur_x); free(sur_y);
    free(fan_x); free(fan_y);

    /* --- Reverse-connect feedback --- */
    for (int bi = 0; bi < total_units; bi++) {
        PVMBlock *b = &pvm->graph[bi];
        for (int di = 0; di < b->n_primary_destinations; di++) {
            int dest = b->primary_destinations[di];
            PVMBlock *db = &pvm->graph[dest];
            int cd_cap = 0;
            append_int(&db->context_destinations, &db->n_context_destinations, &cd_cap, b->id);
            int cs_cap = 0;
            append_int(&b->context_sources, &b->n_context_sources, &cs_cap, dest);
            total_context_projections++;
        }
    }

    /* --- Optional: send_context_two_layers_back --- */
    if (c->send_context_two_layers_back) {
        for (int bi = 0; bi < total_units; bi++) {
            PVMBlock *b = &pvm->graph[bi];
            for (int di = 0; di < b->n_primary_destinations; di++) {
                int dest = b->primary_destinations[di];
                PVMBlock *db = &pvm->graph[dest];
                for (int d2i = 0; d2i < db->n_primary_destinations; d2i++) {
                    int dest1 = db->primary_destinations[d2i];
                    PVMBlock *d1b = &pvm->graph[dest1];
                    int cd_cap = 0;
                    append_int(&d1b->context_destinations, &d1b->n_context_destinations, &cd_cap, b->id);
                    int cs_cap = 0;
                    append_int(&b->context_sources, &b->n_context_sources, &cs_cap, dest1);
                    total_context_projections++;
                }
            }
        }
    }

    /* --- Optional: last_layer_context_to_all --- */
    if (c->last_layer_context_to_all) {
        for (int bi = 0; bi < total_units; bi++) {
            PVMBlock *b = &pvm->graph[bi];
            if (b->layer != num_layers - 1) continue; /* last layer only */
            for (int ui = 0; ui < total_units; ui++) {
                PVMBlock *ub = &pvm->graph[ui];
                if (!id_in_array(b->context_destinations, b->n_context_destinations, ui)) {
                    int cd_cap = 0;
                    append_int(&b->context_destinations, &b->n_context_destinations, &cd_cap, ui);
                    int cs_cap = 0;
                    append_int(&ub->context_sources, &ub->n_context_sources, &cs_cap, bi);
                    total_context_projections++;
                }
            }
        }
    }

    /* --- Compute input_dim, output_dim, context_dim per block --- */
    for (int bi = 0; bi < total_units; bi++) {
        PVMBlock *b = &pvm->graph[bi];
        b->input_dim = b->base_input_offset;
        for (int si = 0; si < b->n_primary_sources; si++) {
            int src_id = b->primary_sources[si];
            b->input_dim += seq_interval * pvm->graph[src_id].size;
            b->output_dim += seq_interval * pvm->graph[src_id].size;
        }
        for (int si = 0; si < b->n_context_sources; si++) {
            int src_id = b->context_sources[si];
            b->input_dim   += pvm->graph[src_id].size;
            b->context_dim += pvm->graph[src_id].size;
        }
    }

    /* --- Compute totals --- */
    int total_weights   = 0;
    int total_input_mem = 0;
    int total_repr_mem  = 0;
    for (int bi = 0; bi < total_units; bi++) {
        PVMBlock *b = &pvm->graph[bi];
        if (c->feed_context_in_complex_layer) {
            total_weights += (b->input_dim + 1) * b->size
                           + (b->size + 1 + b->context_dim) * b->output_dim;
            total_repr_mem += b->size + 1 + b->context_dim;
        } else {
            total_weights += (b->input_dim + 1) * b->size
                           + (b->size + 1) * b->output_dim;
            total_repr_mem += b->size + 1;
        }
        total_input_mem += b->input_dim + 1;
    }

    pvm->total_weights             = total_weights;
    pvm->total_input_mem           = total_input_mem;
    pvm->total_repr_mem            = total_repr_mem;
    pvm->total_primary_projections = total_primary_projections;
    pvm->total_context_projections = total_context_projections;

    printf("Generated connectivity with %d units and %d weights\n", total_units, total_weights);
    printf("Total input mem %d, total representation mem %d\n", total_input_mem, total_repr_mem);
    printf("Total primary projections %d, total context projections %d\n",
           total_primary_projections, total_context_projections);
}

/* =========================================================================
 *  Memory generation
 * ========================================================================= */

void pvm_generate_memory(PVMObject *pvm) {
    int tw = pvm->total_weights;
    int ti = pvm->total_input_mem;
    int tr = pvm->total_repr_mem;

    /* Weights: random init 0.03*(rand - 0.5) */
    pvm->h_weight_mem = (float *)malloc(sizeof(float) * tw);
    srand((unsigned)time(NULL));
    for (int i = 0; i < tw; i++)
        pvm->h_weight_mem[i] = 0.03f * ((float)rand() / (float)RAND_MAX - 0.5f);

    pvm->h_dweight_mem[0]  = (float *)calloc(tw, sizeof(float));
    pvm->h_dweight_mem[1]  = (float *)calloc(tw, sizeof(float));
    pvm->h_weight_cache    = (float *)calloc(tw, sizeof(float));

    pvm->buffer_index = 0;

    /* Input / output / repr memory: activation inited to 1.0, delta/error to 0 */
    for (int s = 0; s < PVM_SEQ_LENGTH; s++) {
        pvm->h_input_activation[s] = (float *)malloc(sizeof(float) * ti);
        for (int j = 0; j < ti; j++) pvm->h_input_activation[s][j] = 1.0f;
        pvm->h_input_delta[s]  = (float *)calloc(ti, sizeof(float));
        pvm->h_input_error[s]  = (float *)calloc(ti, sizeof(float));

        pvm->h_output_activation[s] = (float *)malloc(sizeof(float) * ti);
        for (int j = 0; j < ti; j++) pvm->h_output_activation[s][j] = 1.0f;
        pvm->h_output_delta[s] = (float *)calloc(ti, sizeof(float));
        pvm->h_output_error[s] = (float *)calloc(ti, sizeof(float));

        pvm->h_repr_activation[s] = (float *)malloc(sizeof(float) * tr);
        for (int j = 0; j < tr; j++) pvm->h_repr_activation[s][j] = 1.0f;
        pvm->h_repr_delta[s]  = (float *)calloc(tr, sizeof(float));
        pvm->h_repr_error[s]  = (float *)calloc(tr, sizeof(float));
    }

    /* Beta arrays = 1 */
    pvm->h_beta_input = (float *)malloc(sizeof(float) * ti);
    for (int i = 0; i < ti; i++) pvm->h_beta_input[i] = 1.0f;
    pvm->h_beta_repr = (float *)malloc(sizeof(float) * tr);
    for (int i = 0; i < tr; i++) pvm->h_beta_repr[i] = 1.0f;

    /* Learning-rate / momentum arrays = 0 initially */
    pvm->h_learning_rate_arr = (float *)calloc(pvm->total_units, sizeof(float));
    pvm->h_momentum_arr      = (float *)calloc(pvm->total_units, sizeof(float));
}

/* =========================================================================
 *  Memory pointer generation
 * ========================================================================= */

void pvm_generate_memory_ptrs(PVMObject *pvm) {
    int tu = pvm->total_units;
    PVMConfig *c = &pvm->config;

    pvm->h_weight_ptr0 = (int *)calloc(tu, sizeof(int));
    pvm->h_weight_ptr1 = (int *)calloc(tu, sizeof(int));
    pvm->h_shape0_L0   = (int *)calloc(tu, sizeof(int));
    pvm->h_shape1_L0   = (int *)calloc(tu, sizeof(int));
    pvm->h_shape0_L1   = (int *)calloc(tu, sizeof(int));
    pvm->h_shape1_L1   = (int *)calloc(tu, sizeof(int));
    pvm->h_input_ptr   = (int *)calloc(tu, sizeof(int));
    pvm->h_repr_ptr    = (int *)calloc(tu, sizeof(int));

    int curr_w0_ptr = 0;
    int curr_w1_ptr = 0;
    int curr_i_ptr  = 0;
    int curr_r_ptr  = 0;
    int total_threads_k2_L0 = 0;
    int total_threads_k2_L1 = 0;

    int i = 0;
    for (int bi = 0; bi < tu; bi++) {
        PVMBlock *b = &pvm->graph[bi];

        pvm->h_weight_ptr0[i] = curr_w0_ptr;
        b->w0_ptr = curr_w0_ptr;
        curr_w0_ptr += (b->input_dim + 1) * b->size;
        total_threads_k2_L0 += (b->input_dim + 1);
        pvm->h_shape0_L0[b->id] = b->size;
        pvm->h_shape1_L0[b->id] = b->input_dim + 1;

        pvm->h_weight_ptr1[i] = curr_w1_ptr;
        pvm->h_shape0_L1[b->id] = b->output_dim;
        if (!c->feed_context_in_complex_layer) {
            total_threads_k2_L1 += (b->size + 1);
            pvm->h_shape1_L1[b->id] = b->size + 1;
        } else {
            total_threads_k2_L1 += (b->size + 1 + b->context_dim);
            pvm->h_shape1_L1[b->id] = b->size + 1 + b->context_dim;
        }

        b->w1_ptr = curr_w1_ptr;
        if (!c->feed_context_in_complex_layer)
            curr_w1_ptr += (b->size + 1) * b->output_dim;
        else
            curr_w1_ptr += (b->size + 1 + b->context_dim) * b->output_dim;

        pvm->h_input_ptr[i] = curr_i_ptr;
        b->i_ptr = curr_i_ptr;
        curr_i_ptr += b->input_dim + 1;

        pvm->h_repr_ptr[i] = curr_r_ptr;
        b->r_ptr = curr_r_ptr;
        if (!c->feed_context_in_complex_layer)
            curr_r_ptr += b->size + 1;
        else
            curr_r_ptr += b->size + 1 + b->context_dim;

        i++;
    }

    /* L1 weights come after all L0 weights */
    for (int j = 0; j < tu; j++)
        pvm->h_weight_ptr1[j] += curr_w0_ptr;

    /* Build obj_id and row_id dispatch arrays */
    pvm->h_obj_id_L0 = (int *)calloc(total_threads_k2_L0, sizeof(int));
    pvm->h_row_id_L0 = (int *)calloc(total_threads_k2_L0, sizeof(int));
    pvm->h_obj_id_L1 = (int *)calloc(total_threads_k2_L1, sizeof(int));
    pvm->h_row_id_L1 = (int *)calloc(total_threads_k2_L1, sizeof(int));

    int thread_id_L0 = 0;
    int thread_id_L1 = 0;
    for (int bi = 0; bi < tu; bi++) {
        PVMBlock *b = &pvm->graph[bi];

        /* IMPORTANT: also add curr_w0_ptr offset to the block's w1_ptr */
        b->w1_ptr += curr_w0_ptr;

        for (int j = 0; j < b->input_dim + 1; j++) {
            pvm->h_obj_id_L0[thread_id_L0] = b->id;
            pvm->h_row_id_L0[thread_id_L0] = j;
            thread_id_L0++;
        }
        if (!c->feed_context_in_complex_layer) {
            for (int j = 0; j < b->size + 1; j++) {
                pvm->h_obj_id_L1[thread_id_L1] = b->id;
                pvm->h_row_id_L1[thread_id_L1] = j;
                thread_id_L1++;
            }
        } else {
            for (int j = 0; j < b->size + 1 + b->context_dim; j++) {
                pvm->h_obj_id_L1[thread_id_L1] = b->id;
                pvm->h_row_id_L1[thread_id_L1] = j;
                thread_id_L1++;
            }
        }
    }

    pvm->cuda_block_size = 128;
    pvm->cuda_grid_L0 = total_threads_k2_L0 / pvm->cuda_block_size + 1;
    pvm->cuda_total_threads_k2_L0 = total_threads_k2_L0;
    pvm->cuda_grid_L1 = total_threads_k2_L1 / pvm->cuda_block_size + 1;
    pvm->cuda_total_threads_k2_L1 = total_threads_k2_L1;
}

/* =========================================================================
 *  Flow pointer generation
 * ========================================================================= */

void pvm_generate_flow_ptrs(PVMObject *pvm) {
    int tp = pvm->total_primary_projections;
    int tc = pvm->total_context_projections;
    int tu = pvm->total_units;
    int seq_interval = PVM_SEQ_INTERVAL;
    int ls0 = pvm->config.layer_shapes[0];

    /* Allocate flow arrays */
    pvm->h_flow_from = (int *)calloc(tp + tc, sizeof(int));
    pvm->h_flow_to   = (int *)calloc(tp + tc, sizeof(int));
    pvm->h_flow_size = (int *)calloc(tp + tc, sizeof(int));

    pvm->h_flow_repr_from = (int *)calloc(tc, sizeof(int));
    pvm->h_flow_repr_to   = (int *)calloc(tc, sizeof(int));
    pvm->h_flow_repr_size = (int *)calloc(tc, sizeof(int));

    pvm->h_flow_shift_from = (int *)calloc(tu, sizeof(int));
    pvm->h_flow_shift_to   = (int *)calloc(tu, sizeof(int));
    pvm->h_flow_shift_size = (int *)calloc(tu, sizeof(int));

    pvm->h_flow_input_frame      = (int *)calloc(ls0 * ls0, sizeof(int));
    pvm->h_flow_input_frame_size = (int *)calloc(ls0 * ls0, sizeof(int));

    /* First loop: primary + context flow */
    int fi = 0;
    for (int bi = 0; bi < tu; bi++) {
        PVMBlock *b = &pvm->graph[bi];
        int primary_src_shift = 0;

        for (int pi = 0; pi < b->n_primary_sources; pi++) {
            int src_id = b->primary_sources[pi];
            PVMBlock *src = &pvm->graph[src_id];
            pvm->h_flow_size[fi] = b->size;
            pvm->h_flow_from[fi] = pvm->h_repr_ptr[src_id];
            pvm->h_flow_to[fi]   = b->i_ptr + b->running_input_ptr;
            primary_src_shift    += src->size;
            b->running_input_ptr += src->size;
            fi++;
        }
        b->running_input_ptr += primary_src_shift * (seq_interval - 1);

        for (int ci = 0; ci < b->n_context_sources; ci++) {
            int src_id = b->context_sources[ci];
            PVMBlock *src = &pvm->graph[src_id];
            pvm->h_flow_size[fi] = b->size;
            pvm->h_flow_from[fi] = pvm->h_repr_ptr[src_id];
            pvm->h_flow_to[fi]   = b->i_ptr + b->running_input_ptr;
            b->running_input_ptr += src->size;
            fi++;
        }

        pvm->h_flow_shift_from[b->id] = b->i_ptr;
        pvm->h_flow_shift_to[b->id]   = b->i_ptr + b->base_input_size;
        pvm->h_flow_shift_size[b->id] = b->base_input_size * (seq_interval - 1);
    }

    /* Second loop: repr flow (for feed_context_in_complex_layer) */
    fi = 0;
    for (int bi = 0; bi < tu; bi++) {
        PVMBlock *b = &pvm->graph[bi];
        int running_ptr = 0;
        for (int ci = 0; ci < b->n_context_sources; ci++) {
            int src_id = b->context_sources[ci];
            PVMBlock *src = &pvm->graph[src_id];
            pvm->h_flow_repr_size[fi] = src->size;
            pvm->h_flow_repr_from[fi] = pvm->h_repr_ptr[src_id];
            pvm->h_flow_repr_to[fi]   = b->r_ptr + b->size + 1 + running_ptr;
            running_ptr += src->size;
            fi++;
        }
    }

    /* Third part: input frame pointers for layer 0 */
    for (int j = 0; j < ls0 * ls0; j++) {
        pvm->h_flow_input_frame[j]      = pvm->graph[j].i_ptr;
        pvm->h_flow_input_frame_size[j] = pvm->graph[j].base_input_size;
    }
}

/* =========================================================================
 *  GPU memory creation
 * ========================================================================= */

/* Helper macro for cudaMalloc + cudaMemcpy */
#define GPU_ALLOC_COPY_FLOAT(d_ptr, h_ptr, count) do {                        \
    cudaMalloc((void **)&(d_ptr), sizeof(float) * (count));                   \
    cudaMemcpy((d_ptr), (h_ptr), sizeof(float) * (count), cudaMemcpyHostToDevice); \
} while(0)

#define GPU_ALLOC_COPY_INT(d_ptr, h_ptr, count) do {                          \
    cudaMalloc((void **)&(d_ptr), sizeof(int) * (count));                     \
    cudaMemcpy((d_ptr), (h_ptr), sizeof(int) * (count), cudaMemcpyHostToDevice); \
} while(0)

void pvm_create_gpu_mem(PVMObject *pvm) {
    int tw = pvm->total_weights;
    int ti = pvm->total_input_mem;
    int tr = pvm->total_repr_mem;
    int tu = pvm->total_units;
    int tp = pvm->total_primary_projections;
    int tc = pvm->total_context_projections;
    int ls0 = pvm->config.layer_shapes[0];

    /* Float arrays */
    GPU_ALLOC_COPY_FLOAT(pvm->d_weight_mem,     pvm->h_weight_mem,     tw);
    GPU_ALLOC_COPY_FLOAT(pvm->d_dweight_mem[0],  pvm->h_dweight_mem[0],  tw);
    GPU_ALLOC_COPY_FLOAT(pvm->d_dweight_mem[1],  pvm->h_dweight_mem[1],  tw);
    GPU_ALLOC_COPY_FLOAT(pvm->d_weight_cache,    pvm->h_weight_cache,    tw);

    for (int s = 0; s < PVM_SEQ_LENGTH; s++) {
        GPU_ALLOC_COPY_FLOAT(pvm->d_input_activation[s],  pvm->h_input_activation[s],  ti);
        GPU_ALLOC_COPY_FLOAT(pvm->d_input_delta[s],       pvm->h_input_delta[s],       ti);
        GPU_ALLOC_COPY_FLOAT(pvm->d_input_error[s],       pvm->h_input_error[s],       ti);

        GPU_ALLOC_COPY_FLOAT(pvm->d_output_activation[s], pvm->h_output_activation[s], ti);
        GPU_ALLOC_COPY_FLOAT(pvm->d_output_delta[s],      pvm->h_output_delta[s],      ti);
        GPU_ALLOC_COPY_FLOAT(pvm->d_output_error[s],      pvm->h_output_error[s],      ti);

        GPU_ALLOC_COPY_FLOAT(pvm->d_repr_activation[s],   pvm->h_repr_activation[s],   tr);
        GPU_ALLOC_COPY_FLOAT(pvm->d_repr_delta[s],        pvm->h_repr_delta[s],        tr);
        GPU_ALLOC_COPY_FLOAT(pvm->d_repr_error[s],        pvm->h_repr_error[s],        tr);
    }

    GPU_ALLOC_COPY_FLOAT(pvm->d_beta_input,         pvm->h_beta_input,         ti);
    GPU_ALLOC_COPY_FLOAT(pvm->d_beta_repr,          pvm->h_beta_repr,          tr);
    GPU_ALLOC_COPY_FLOAT(pvm->d_learning_rate_arr,  pvm->h_learning_rate_arr,  tu);
    GPU_ALLOC_COPY_FLOAT(pvm->d_momentum_arr,       pvm->h_momentum_arr,       tu);

    /* Int arrays -- pointer / shape arrays */
    GPU_ALLOC_COPY_INT(pvm->d_weight_ptr0,  pvm->h_weight_ptr0,  tu);
    GPU_ALLOC_COPY_INT(pvm->d_weight_ptr1,  pvm->h_weight_ptr1,  tu);
    GPU_ALLOC_COPY_INT(pvm->d_input_ptr,    pvm->h_input_ptr,    tu);
    GPU_ALLOC_COPY_INT(pvm->d_repr_ptr,     pvm->h_repr_ptr,     tu);
    GPU_ALLOC_COPY_INT(pvm->d_shape0_L0,    pvm->h_shape0_L0,    tu);
    GPU_ALLOC_COPY_INT(pvm->d_shape1_L0,    pvm->h_shape1_L0,    tu);
    GPU_ALLOC_COPY_INT(pvm->d_shape0_L1,    pvm->h_shape0_L1,    tu);
    GPU_ALLOC_COPY_INT(pvm->d_shape1_L1,    pvm->h_shape1_L1,    tu);

    GPU_ALLOC_COPY_INT(pvm->d_obj_id_L0,    pvm->h_obj_id_L0,    pvm->cuda_total_threads_k2_L0);
    GPU_ALLOC_COPY_INT(pvm->d_row_id_L0,    pvm->h_row_id_L0,    pvm->cuda_total_threads_k2_L0);
    GPU_ALLOC_COPY_INT(pvm->d_obj_id_L1,    pvm->h_obj_id_L1,    pvm->cuda_total_threads_k2_L1);
    GPU_ALLOC_COPY_INT(pvm->d_row_id_L1,    pvm->h_row_id_L1,    pvm->cuda_total_threads_k2_L1);

    /* Flow pointers */
    GPU_ALLOC_COPY_INT(pvm->d_flow_from,     pvm->h_flow_from,     tp + tc);
    GPU_ALLOC_COPY_INT(pvm->d_flow_to,       pvm->h_flow_to,       tp + tc);
    GPU_ALLOC_COPY_INT(pvm->d_flow_size,     pvm->h_flow_size,     tp + tc);

    GPU_ALLOC_COPY_INT(pvm->d_flow_repr_from, pvm->h_flow_repr_from, tc);
    GPU_ALLOC_COPY_INT(pvm->d_flow_repr_to,   pvm->h_flow_repr_to,   tc);
    GPU_ALLOC_COPY_INT(pvm->d_flow_repr_size, pvm->h_flow_repr_size, tc);

    GPU_ALLOC_COPY_INT(pvm->d_flow_shift_from, pvm->h_flow_shift_from, tu);
    GPU_ALLOC_COPY_INT(pvm->d_flow_shift_to,   pvm->h_flow_shift_to,   tu);
    GPU_ALLOC_COPY_INT(pvm->d_flow_shift_size, pvm->h_flow_shift_size, tu);

    GPU_ALLOC_COPY_INT(pvm->d_flow_input_frame,      pvm->h_flow_input_frame,      ls0 * ls0);
    GPU_ALLOC_COPY_INT(pvm->d_flow_input_frame_size,  pvm->h_flow_input_frame_size, ls0 * ls0);
}

#undef GPU_ALLOC_COPY_FLOAT
#undef GPU_ALLOC_COPY_INT

/* =========================================================================
 *  Object creation
 * ========================================================================= */

PVMObject* pvm_object_create(const PVMConfig *config, const char *name) {
    PVMObject *pvm = (PVMObject *)calloc(1, sizeof(PVMObject));

    /* Copy config */
    memcpy(&pvm->config, config, sizeof(PVMConfig));

    /* Defaults */
    pvm->step          = 0;
    pvm->buffer_index  = 0;

    /* Input channels from config (default 3) */
    pvm->input_channels = config->input_channels > 0 ? config->input_channels : 3;

    /* Polynomial activation */
    pvm->poly = config->polynomial;

    /* Name */
    if (name)
        snprintf(pvm->name, sizeof(pvm->name), "%s", name);
    else
        snprintf(pvm->name, sizeof(pvm->name), "noname");

    /* Unique ID: random 8 hex digits */
    srand((unsigned)time(NULL));
    snprintf(pvm->uniq_id, sizeof(pvm->uniq_id), "%08x", (unsigned)rand());

    /* Timestamp */
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(pvm->time_stamp, sizeof(pvm->time_stamp), "%Y_%m_%d_%H_%M_%S", tm_info);

    /* CUDA device name */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    snprintf(pvm->device_name, sizeof(pvm->device_name), "%s", prop.name);

    /* Saved lr/momentum pointers init to NULL */
    pvm->h_learning_rate_saved = NULL;
    pvm->h_momentum_saved      = NULL;

    /* Build everything */
    pvm_generate_graph(pvm);
    pvm_generate_memory(pvm);
    pvm_generate_memory_ptrs(pvm);
    pvm_generate_flow_ptrs(pvm);
    pvm_create_gpu_mem(pvm);

    return pvm;
}

/* =========================================================================
 *  Forward pass
 * ========================================================================= */

void pvm_forward(PVMObject *pvm) {
    int cs = pvm->step % PVM_SEQ_LENGTH;
    int tu = pvm->total_units;
    int bs = pvm->cuda_block_size;

    /* Zero repr activation for current step */
    gpu_memset_zero(pvm->d_repr_activation[cs], sizeof(float) * pvm->total_repr_mem);

    /* L0: weight_mem * input_act -> repr_act */
    gpu_dot_fast_set_bias<<<pvm->cuda_grid_L0, bs>>>(
        pvm->d_weight_mem,
        pvm->d_input_activation[cs],
        pvm->d_repr_activation[cs],
        pvm->d_weight_ptr0,
        pvm->d_input_ptr,
        pvm->d_repr_ptr,
        pvm->d_shape0_L0,
        pvm->d_shape1_L0,
        pvm->d_obj_id_L0,
        pvm->d_row_id_L0,
        pvm->cuda_total_threads_k2_L0);

    /* Sigmoid on repr_act */
    if (!pvm->poly) {
        gpu_sigmoid_fast<<<tu / 512 + 1, 512>>>(
            pvm->d_repr_activation[cs],
            pvm->d_repr_ptr,
            pvm->d_beta_repr,
            pvm->d_shape0_L0,
            tu);
    } else {
        gpu_sigmoid_poly_fast<<<tu / 512 + 1, 512>>>(
            pvm->d_repr_activation[cs],
            pvm->d_repr_ptr,
            pvm->d_beta_repr,
            pvm->d_shape0_L0,
            tu);
    }

    /* Zero output activation for current step */
    gpu_memset_zero(pvm->d_output_activation[cs], sizeof(float) * pvm->total_input_mem);

    /* L1: weight_mem * repr_act -> output_act */
    gpu_dot_fast_set_bias<<<pvm->cuda_grid_L1, bs>>>(
        pvm->d_weight_mem,
        pvm->d_repr_activation[cs],
        pvm->d_output_activation[cs],
        pvm->d_weight_ptr1,
        pvm->d_repr_ptr,
        pvm->d_input_ptr,
        pvm->d_shape0_L1,
        pvm->d_shape1_L1,
        pvm->d_obj_id_L1,
        pvm->d_row_id_L1,
        pvm->cuda_total_threads_k2_L1);

    /* Sigmoid on output_act */
    if (!pvm->poly) {
        gpu_sigmoid_fast<<<tu / 512 + 1, 512>>>(
            pvm->d_output_activation[cs],
            pvm->d_input_ptr,
            pvm->d_beta_input,
            pvm->d_shape0_L1,
            tu);
    } else {
        gpu_sigmoid_poly_fast<<<tu / 512 + 1, 512>>>(
            pvm->d_output_activation[cs],
            pvm->d_input_ptr,
            pvm->d_beta_input,
            pvm->d_shape0_L1,
            tu);
    }
}

/* =========================================================================
 *  Backward pass
 * ========================================================================= */

void pvm_backward(PVMObject *pvm) {
    int seq_interval = PVM_SEQ_INTERVAL;
    /* Safe modulo for possibly negative operand */
    int as = ((pvm->step - seq_interval) % PVM_SEQ_LENGTH + PVM_SEQ_LENGTH) % PVM_SEQ_LENGTH;
    int cs = pvm->step % PVM_SEQ_LENGTH;
    int tu = pvm->total_units;
    int ti = pvm->total_input_mem;
    int bs = pvm->cuda_block_size;

    /* Compute output error: error = input_act[cs] - output_act[as] */
    gpu_memset_zero(pvm->d_output_error[as], sizeof(float) * ti);

    /* error = 0 - output_act[as] */
    cudaMemcpy(pvm->d_output_error[as], pvm->d_output_activation[as],
               sizeof(float) * ti, cudaMemcpyDeviceToDevice);
    /* error = input_act[cs] - error (i.e. input_act[cs] - output_act[as]) */
    axpy_neg<<<(ti + 511) / 512, 512>>>(
        pvm->d_input_activation[cs],
        pvm->d_output_error[as],
        ti);

    /* Optional: sign of error for abs diff */
    if (pvm->config.opt_abs_diff) {
        gpu_sgn<<<tu / 512 + 1, 512>>>(
            pvm->d_output_error[as],
            tu);
    }

    /* Derivative of output layer */
    if (!pvm->poly) {
        gpu_sigmoid_der_mul<<<tu / 512 + 1, 512>>>(
            pvm->d_output_activation[as],
            pvm->d_output_error[as],
            pvm->d_output_delta[as],
            pvm->d_input_ptr,
            pvm->d_input_ptr,
            pvm->d_input_ptr,
            pvm->d_shape0_L1,
            tu);
    } else {
        gpu_sigmoid_poly_der_mul<<<tu / 512 + 1, 512>>>(
            pvm->d_output_activation[as],
            pvm->d_output_error[as],
            pvm->d_output_delta[as],
            pvm->d_input_ptr,
            pvm->d_input_ptr,
            pvm->d_input_ptr,
            pvm->d_shape0_L1,
            tu);
    }

    /* Backpropagate to representation layer */
    gpu_dot_transpose_fast<<<pvm->cuda_grid_L1, bs>>>(
        pvm->d_weight_mem,
        pvm->d_weight_cache,
        pvm->d_output_delta[as],
        pvm->d_weight_ptr1,
        pvm->d_input_ptr,
        pvm->d_shape0_L1,
        pvm->d_shape1_L1,
        pvm->d_obj_id_L1,
        pvm->d_row_id_L1,
        pvm->cuda_total_threads_k2_L1);

    gpu_memset_zero(pvm->d_repr_error[as], sizeof(float) * pvm->total_repr_mem);

    gpu_sum_dot_transpose<<<pvm->cuda_grid_L1, bs>>>(
        pvm->d_weight_cache,
        pvm->d_repr_error[as],
        pvm->d_weight_ptr1,
        pvm->d_repr_ptr,
        pvm->d_shape0_L1,
        pvm->d_shape1_L1,
        pvm->d_obj_id_L1,
        pvm->d_row_id_L1,
        pvm->cuda_total_threads_k2_L1);

    /* Derivative in the representation layer */
    if (!pvm->poly) {
        gpu_sigmoid_der_mul<<<tu / 512 + 1, 512>>>(
            pvm->d_repr_activation[as],
            pvm->d_repr_error[as],
            pvm->d_repr_delta[as],
            pvm->d_repr_ptr,
            pvm->d_repr_ptr,
            pvm->d_repr_ptr,
            pvm->d_shape1_L1,
            tu);
    } else {
        gpu_sigmoid_poly_der_mul<<<tu / 512 + 1, 512>>>(
            pvm->d_repr_activation[as],
            pvm->d_repr_error[as],
            pvm->d_repr_delta[as],
            pvm->d_repr_ptr,
            pvm->d_repr_ptr,
            pvm->d_repr_ptr,
            pvm->d_shape1_L1,
            tu);
    }

    /* Weight update for L1: outer product (output_delta x repr_activation) */
    gpu_generalized_outer_fast3<<<pvm->cuda_grid_L1, bs>>>(
        pvm->d_output_delta[as],
        pvm->d_repr_activation[as],
        pvm->d_dweight_mem[(pvm->buffer_index + 1) % 2],
        pvm->d_dweight_mem[pvm->buffer_index],
        pvm->d_input_ptr,
        pvm->d_repr_ptr,
        pvm->d_weight_ptr1,
        pvm->d_weight_ptr1,
        pvm->d_shape0_L1,
        pvm->d_shape1_L1,
        pvm->d_learning_rate_arr,
        pvm->d_momentum_arr,
        pvm->d_obj_id_L1,
        pvm->d_row_id_L1,
        pvm->cuda_total_threads_k2_L1);

    /* Weight update for L0: outer product (repr_delta x input_activation) */
    gpu_generalized_outer_fast3<<<pvm->cuda_grid_L0, bs>>>(
        pvm->d_repr_delta[as],
        pvm->d_input_activation[as],
        pvm->d_dweight_mem[(pvm->buffer_index + 1) % 2],
        pvm->d_dweight_mem[pvm->buffer_index],
        pvm->d_repr_ptr,
        pvm->d_input_ptr,
        pvm->d_weight_ptr0,
        pvm->d_weight_ptr0,
        pvm->d_shape0_L0,
        pvm->d_shape1_L0,
        pvm->d_learning_rate_arr,
        pvm->d_momentum_arr,
        pvm->d_obj_id_L0,
        pvm->d_row_id_L0,
        pvm->cuda_total_threads_k2_L0);

    /* Apply weight update: weight_mem += dweight[buffer_index] */
    int tw = pvm->total_weights;
    axpy_add<<<(tw + 511) / 512, 512>>>(
        pvm->d_dweight_mem[pvm->buffer_index],
        pvm->d_weight_mem,
        tw);

    /* Flip buffer */
    pvm->buffer_index = (pvm->buffer_index + 1) % 2;
}

/* =========================================================================
 *  Push input
 * ========================================================================= */

void pvm_push_input(PVMObject *pvm, const float *frame, int h, int w, int ch) {
    int tu = pvm->total_units;
    int tp = pvm->total_primary_projections;
    int tc = pvm->total_context_projections;
    int total_projections = tp + tc;
    int seq_length = PVM_SEQ_LENGTH;
    int ls0 = pvm->config.layer_shapes[0];
    int ibs = pvm->config.input_block_size;

    /* Upload frame to GPU (temporary) */
    int frame_elems = h * w * ch;
    float *d_frame = NULL;
    cudaMalloc((void **)&d_frame, sizeof(float) * frame_elems);
    cudaMemcpy(d_frame, frame, sizeof(float) * frame_elems, cudaMemcpyHostToDevice);

    int prev_step = ((pvm->step - 1) % seq_length + seq_length) % seq_length;
    int curr_step = pvm->step % seq_length;

    /* Shift inputs: copy from previous step's activation to current step */
    gpu_copy_blocks<<<tu / 128 + 1, (tu < 128 ? tu : 128)>>>(
        pvm->d_input_activation[prev_step],
        pvm->d_input_activation[curr_step],
        pvm->d_flow_shift_from,
        pvm->d_flow_shift_size,
        pvm->d_flow_shift_to,
        tu);

    /* Zero repr activation for current step */
    gpu_memset_zero(pvm->d_repr_activation[curr_step], sizeof(float) * pvm->total_repr_mem);

    /* Copy primary + context: compressed copy */
    if (total_projections > 0) {
        gpu_copy_blocks_comp<<<total_projections / 128 + 1,
                               (total_projections < 128 ? total_projections : 128)>>>(
            pvm->d_repr_activation[prev_step],
            pvm->d_input_activation[curr_step],
            pvm->d_flow_from,
            pvm->d_flow_size,
            pvm->d_flow_to,
            total_projections);
    }

    /* Optional: feed context in complex layer */
    if (pvm->config.feed_context_in_complex_layer && tc > 0) {
        gpu_copy_blocks<<<tc / 128 + 1, (tc < 128 ? tc : 128)>>>(
            pvm->d_repr_activation[prev_step],
            pvm->d_repr_activation[curr_step],
            pvm->d_flow_repr_from,
            pvm->d_flow_repr_size,
            pvm->d_flow_repr_to,
            tc);
    }

    /* Distribute frame to input blocks */
    int frame_patches = ls0 * ls0;
    int frame_dim = ls0 * ibs;
    if (ch == 3) {
        gpu_dist_frame<<<frame_patches / 128 + 1, (frame_patches < 128 ? frame_patches : 128)>>>(
            d_frame,
            pvm->d_input_activation[curr_step],
            pvm->d_flow_input_frame,
            frame_dim, frame_dim,
            ls0, ls0,
            ibs, ibs,
            0,
            frame_patches);
    } else if (ch == 4) {
        gpu_dist_frame4<<<frame_patches / 128 + 1, (frame_patches < 128 ? frame_patches : 128)>>>(
            d_frame,
            pvm->d_input_activation[curr_step],
            pvm->d_flow_input_frame,
            frame_dim, frame_dim,
            ls0, ls0,
            ibs, ibs,
            0,
            frame_patches);
    }

    /* Free temporary frame */
    cudaFree(d_frame);
}

/* =========================================================================
 *  Pop prediction
 * ========================================================================= */

void pvm_pop_prediction(PVMObject *pvm, float *out_buf, int delta_step) {
    int ls0 = pvm->config.layer_shapes[0];
    int ibs = pvm->config.input_block_size;
    int ch  = pvm->input_channels;
    int frame_dim = ls0 * ibs;
    int frame_elems = frame_dim * frame_dim * ch;
    int frame_patches = ls0 * ls0;
    int step_idx = ((pvm->step + delta_step) % PVM_SEQ_LENGTH + PVM_SEQ_LENGTH) % PVM_SEQ_LENGTH;

    float *d_frame = NULL;
    cudaMalloc((void **)&d_frame, sizeof(float) * frame_elems);
    cudaMemset(d_frame, 0, sizeof(float) * frame_elems);

    if (ch == 3) {
        gpu_collect_frame<<<frame_patches / 128 + 1, (frame_patches < 128 ? frame_patches : 128)>>>(
            d_frame,
            pvm->d_output_activation[step_idx],
            pvm->d_flow_input_frame,
            frame_dim, frame_dim,
            ls0, ls0,
            ibs, ibs,
            0,
            frame_patches);
    } else if (ch == 4) {
        gpu_collect_frame4<<<frame_patches / 128 + 1, (frame_patches < 128 ? frame_patches : 128)>>>(
            d_frame,
            pvm->d_output_activation[step_idx],
            pvm->d_flow_input_frame,
            frame_dim, frame_dim,
            ls0, ls0,
            ibs, ibs,
            0,
            frame_patches);
    }

    cudaMemcpy(out_buf, d_frame, sizeof(float) * frame_elems, cudaMemcpyDeviceToHost);
    cudaFree(d_frame);
}

/* =========================================================================
 *  Pop layer
 * ========================================================================= */

void pvm_pop_layer(PVMObject *pvm, unsigned char *out_buf, int layer) {
    int ls_l = pvm->config.layer_shapes[layer];
    int hbs  = pvm->config.hidden_block_size;
    int dim  = ls_l * hbs;
    int frame_patches = ls_l * ls_l;
    int cs = pvm->step % PVM_SEQ_LENGTH;

    unsigned int *d_frame = NULL;
    cudaMalloc((void **)&d_frame, sizeof(unsigned int) * dim * dim);
    cudaMemset(d_frame, 0, sizeof(unsigned int) * dim * dim);

    gpu_collect_activ<<<frame_patches / 128 + 1, (frame_patches < 128 ? frame_patches : 128)>>>(
        d_frame,
        pvm->d_repr_activation[cs],
        pvm->d_repr_ptr,
        dim, dim,
        ls_l, ls_l,
        hbs, hbs,
        pvm->layer_ptrs[layer],
        frame_patches);

    unsigned int *h_frame = (unsigned int *)malloc(sizeof(unsigned int) * dim * dim);
    cudaMemcpy(h_frame, d_frame, sizeof(unsigned int) * dim * dim, cudaMemcpyDeviceToHost);

    for (int i = 0; i < dim * dim; i++)
        out_buf[i] = (unsigned char)(h_frame[i] > 255 ? 255 : h_frame[i]);

    free(h_frame);
    cudaFree(d_frame);
}

/* =========================================================================
 *  Update learning rate
 * ========================================================================= */

void pvm_update_learning_rate(PVMObject *pvm, float override_rate) {
    PVMConfig *c = &pvm->config;
    int *ls = c->layer_shapes;
    int num_layers = c->num_layers;

    /* Possibly enable next layer */
    if (c->delay_each_layer_learning > 0 && pvm->step % c->delay_each_layer_learning == 0) {
        int layer_to_enable = pvm->step / c->delay_each_layer_learning;
        if (layer_to_enable < num_layers) {
            int begin_idx = 0;
            for (int l = 0; l < layer_to_enable; l++)
                begin_idx += ls[l] * ls[l];
            int end_idx = begin_idx + ls[layer_to_enable] * ls[layer_to_enable];
            printf("\nEnabling layer %d\n", layer_to_enable);
            printf("Begin idx %d end idx %d\n", begin_idx, end_idx);
            for (int j = begin_idx; j < end_idx; j++) {
                pvm->h_learning_rate_arr[j] = c->initial_learning_rate;
                pvm->h_momentum_arr[j]      = c->momentum_val;
            }
            pvm->learning_rate = c->initial_learning_rate;
        }
    }

    /* Final learning rate */
    if (pvm->step == c->delay_final_learning_rate) {
        for (int j = 0; j < pvm->total_units; j++)
            pvm->h_learning_rate_arr[j] = c->final_learning_rate;
        pvm->learning_rate = c->final_learning_rate;
        printf("Setting final learning rate\n");
    }

    /* Intermediate learning rate */
    if (c->delay_intermediate_learning_rate > 0 && pvm->step == c->delay_intermediate_learning_rate) {
        for (int j = 0; j < pvm->total_units; j++)
            pvm->h_learning_rate_arr[j] = c->intermediate_learning_rate;
        pvm->learning_rate = c->intermediate_learning_rate;
        printf("Setting intermediate learning rate\n");
    }

    /* Override */
    if (override_rate >= 0.0f) {
        for (int j = 0; j < pvm->total_units; j++)
            pvm->h_learning_rate_arr[j] = override_rate;
        pvm->learning_rate = override_rate;
        printf("Overriding PVM learning rate to %f\n", override_rate);
    }

    /* Upload to GPU */
    cudaMemcpy(pvm->d_learning_rate_arr, pvm->h_learning_rate_arr,
               sizeof(float) * pvm->total_units, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_momentum_arr, pvm->h_momentum_arr,
               sizeof(float) * pvm->total_units, cudaMemcpyHostToDevice);
}

/* =========================================================================
 *  Freeze / unfreeze learning
 * ========================================================================= */

void pvm_freeze_learning(PVMObject *pvm) {
    int tu = pvm->total_units;
    if (pvm->h_learning_rate_saved == NULL)
        pvm->h_learning_rate_saved = (float *)malloc(sizeof(float) * tu);
    if (pvm->h_momentum_saved == NULL)
        pvm->h_momentum_saved = (float *)malloc(sizeof(float) * tu);

    memcpy(pvm->h_learning_rate_saved, pvm->h_learning_rate_arr, sizeof(float) * tu);
    memcpy(pvm->h_momentum_saved, pvm->h_momentum_arr, sizeof(float) * tu);

    memset(pvm->h_learning_rate_arr, 0, sizeof(float) * tu);
    memset(pvm->h_momentum_arr, 0, sizeof(float) * tu);

    cudaMemcpy(pvm->d_learning_rate_arr, pvm->h_learning_rate_arr,
               sizeof(float) * tu, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_momentum_arr, pvm->h_momentum_arr,
               sizeof(float) * tu, cudaMemcpyHostToDevice);
}

void pvm_unfreeze_learning(PVMObject *pvm) {
    int tu = pvm->total_units;
    if (pvm->h_learning_rate_saved != NULL)
        memcpy(pvm->h_learning_rate_arr, pvm->h_learning_rate_saved, sizeof(float) * tu);
    if (pvm->h_momentum_saved != NULL)
        memcpy(pvm->h_momentum_arr, pvm->h_momentum_saved, sizeof(float) * tu);

    cudaMemcpy(pvm->d_learning_rate_arr, pvm->h_learning_rate_arr,
               sizeof(float) * tu, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_momentum_arr, pvm->h_momentum_arr,
               sizeof(float) * tu, cudaMemcpyHostToDevice);
}

/* =========================================================================
 *  Get data from GPU
 * ========================================================================= */

void pvm_get_data_from_gpu(PVMObject *pvm) {
    int tw = pvm->total_weights;
    int ti = pvm->total_input_mem;
    int tr = pvm->total_repr_mem;

    cudaMemcpy(pvm->h_weight_mem, pvm->d_weight_mem,
               sizeof(float) * tw, cudaMemcpyDeviceToHost);
    cudaMemcpy(pvm->h_dweight_mem[0], pvm->d_dweight_mem[0],
               sizeof(float) * tw, cudaMemcpyDeviceToHost);
    cudaMemcpy(pvm->h_dweight_mem[1], pvm->d_dweight_mem[1],
               sizeof(float) * tw, cudaMemcpyDeviceToHost);
    cudaMemcpy(pvm->h_weight_cache, pvm->d_weight_cache,
               sizeof(float) * tw, cudaMemcpyDeviceToHost);

    for (int s = 0; s < PVM_SEQ_LENGTH; s++) {
        cudaMemcpy(pvm->h_input_activation[s], pvm->d_input_activation[s],
                   sizeof(float) * ti, cudaMemcpyDeviceToHost);
        cudaMemcpy(pvm->h_input_delta[s], pvm->d_input_delta[s],
                   sizeof(float) * ti, cudaMemcpyDeviceToHost);
        cudaMemcpy(pvm->h_input_error[s], pvm->d_input_error[s],
                   sizeof(float) * ti, cudaMemcpyDeviceToHost);

        cudaMemcpy(pvm->h_output_activation[s], pvm->d_output_activation[s],
                   sizeof(float) * ti, cudaMemcpyDeviceToHost);
        cudaMemcpy(pvm->h_output_delta[s], pvm->d_output_delta[s],
                   sizeof(float) * ti, cudaMemcpyDeviceToHost);
        cudaMemcpy(pvm->h_output_error[s], pvm->d_output_error[s],
                   sizeof(float) * ti, cudaMemcpyDeviceToHost);

        cudaMemcpy(pvm->h_repr_activation[s], pvm->d_repr_activation[s],
                   sizeof(float) * tr, cudaMemcpyDeviceToHost);
        cudaMemcpy(pvm->h_repr_delta[s], pvm->d_repr_delta[s],
                   sizeof(float) * tr, cudaMemcpyDeviceToHost);
        cudaMemcpy(pvm->h_repr_error[s], pvm->d_repr_error[s],
                   sizeof(float) * tr, cudaMemcpyDeviceToHost);
    }
}

/* =========================================================================
 *  Get input shape
 * ========================================================================= */

void pvm_get_input_shape(PVMObject *pvm, int *w, int *h, int *ch) {
    int s1 = pvm->config.layer_shapes[0] * pvm->config.input_block_size;
    *w  = s1;
    *h  = s1;
    *ch = pvm->input_channels;
}

/* =========================================================================
 *  Save
 * ========================================================================= */

#define PVM_MAGIC "PVM1"

static void write_buf(FILE *fp, const void *buf, size_t nbytes) {
    fwrite(buf, 1, nbytes, fp);
}

static void read_buf(FILE *fp, void *buf, size_t nbytes) {
    size_t r = fread(buf, 1, nbytes, fp);
    (void)r;
}

int pvm_save(PVMObject *pvm, const char *filename) {
    pvm_get_data_from_gpu(pvm);

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "pvm_save: cannot open %s\n", filename);
        return -1;
    }

    int tw = pvm->total_weights;
    int ti = pvm->total_input_mem;
    int tr = pvm->total_repr_mem;

    /* Magic */
    write_buf(fp, PVM_MAGIC, 4);

    /* Config */
    write_buf(fp, &pvm->config, sizeof(PVMConfig));

    /* Step and buffer index */
    write_buf(fp, &pvm->step, sizeof(int));
    write_buf(fp, &pvm->buffer_index, sizeof(int));

    /* Name, id, timestamp, device */
    write_buf(fp, pvm->name, sizeof(pvm->name));
    write_buf(fp, pvm->uniq_id, sizeof(pvm->uniq_id));
    write_buf(fp, pvm->time_stamp, sizeof(pvm->time_stamp));
    write_buf(fp, pvm->device_name, sizeof(pvm->device_name));

    /* Float arrays */
    write_buf(fp, pvm->h_weight_mem, sizeof(float) * tw);
    write_buf(fp, pvm->h_dweight_mem[0], sizeof(float) * tw);
    write_buf(fp, pvm->h_dweight_mem[1], sizeof(float) * tw);
    write_buf(fp, pvm->h_weight_cache, sizeof(float) * tw);

    for (int s = 0; s < PVM_SEQ_LENGTH; s++) {
        write_buf(fp, pvm->h_input_activation[s], sizeof(float) * ti);
        write_buf(fp, pvm->h_input_delta[s],      sizeof(float) * ti);
        write_buf(fp, pvm->h_input_error[s],       sizeof(float) * ti);
        write_buf(fp, pvm->h_output_activation[s], sizeof(float) * ti);
        write_buf(fp, pvm->h_output_delta[s],      sizeof(float) * ti);
        write_buf(fp, pvm->h_output_error[s],      sizeof(float) * ti);
        write_buf(fp, pvm->h_repr_activation[s],   sizeof(float) * tr);
        write_buf(fp, pvm->h_repr_delta[s],        sizeof(float) * tr);
        write_buf(fp, pvm->h_repr_error[s],        sizeof(float) * tr);
    }

    write_buf(fp, pvm->h_beta_input,        sizeof(float) * ti);
    write_buf(fp, pvm->h_beta_repr,         sizeof(float) * tr);
    write_buf(fp, pvm->h_learning_rate_arr, sizeof(float) * pvm->total_units);
    write_buf(fp, pvm->h_momentum_arr,      sizeof(float) * pvm->total_units);

    fclose(fp);
    printf("Saved PVM model to %s\n", filename);
    return 0;
}

/* =========================================================================
 *  Load
 * ========================================================================= */

PVMObject* pvm_load(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "pvm_load: cannot open %s\n", filename);
        return NULL;
    }

    /* Magic */
    char magic[4];
    read_buf(fp, magic, 4);
    if (memcmp(magic, PVM_MAGIC, 4) != 0) {
        fprintf(stderr, "pvm_load: bad magic\n");
        fclose(fp);
        return NULL;
    }

    /* Read config */
    PVMConfig config;
    read_buf(fp, &config, sizeof(PVMConfig));

    int step, buffer_index;
    read_buf(fp, &step, sizeof(int));
    read_buf(fp, &buffer_index, sizeof(int));

    /* Create object from config (this builds graph + allocates memory) */
    PVMObject *pvm = pvm_object_create(&config, NULL);
    pvm->step         = step;
    pvm->buffer_index = buffer_index;

    /* Read name, id, timestamp, device */
    read_buf(fp, pvm->name, sizeof(pvm->name));
    read_buf(fp, pvm->uniq_id, sizeof(pvm->uniq_id));
    read_buf(fp, pvm->time_stamp, sizeof(pvm->time_stamp));
    read_buf(fp, pvm->device_name, sizeof(pvm->device_name));

    int tw = pvm->total_weights;
    int ti = pvm->total_input_mem;
    int tr = pvm->total_repr_mem;

    /* Override float arrays from file */
    read_buf(fp, pvm->h_weight_mem, sizeof(float) * tw);
    read_buf(fp, pvm->h_dweight_mem[0], sizeof(float) * tw);
    read_buf(fp, pvm->h_dweight_mem[1], sizeof(float) * tw);
    read_buf(fp, pvm->h_weight_cache, sizeof(float) * tw);

    for (int s = 0; s < PVM_SEQ_LENGTH; s++) {
        read_buf(fp, pvm->h_input_activation[s], sizeof(float) * ti);
        read_buf(fp, pvm->h_input_delta[s],      sizeof(float) * ti);
        read_buf(fp, pvm->h_input_error[s],       sizeof(float) * ti);
        read_buf(fp, pvm->h_output_activation[s], sizeof(float) * ti);
        read_buf(fp, pvm->h_output_delta[s],      sizeof(float) * ti);
        read_buf(fp, pvm->h_output_error[s],      sizeof(float) * ti);
        read_buf(fp, pvm->h_repr_activation[s],   sizeof(float) * tr);
        read_buf(fp, pvm->h_repr_delta[s],        sizeof(float) * tr);
        read_buf(fp, pvm->h_repr_error[s],        sizeof(float) * tr);
    }

    read_buf(fp, pvm->h_beta_input,        sizeof(float) * ti);
    read_buf(fp, pvm->h_beta_repr,         sizeof(float) * tr);
    read_buf(fp, pvm->h_learning_rate_arr, sizeof(float) * pvm->total_units);
    read_buf(fp, pvm->h_momentum_arr,      sizeof(float) * pvm->total_units);

    fclose(fp);

    /* Re-upload all data to GPU */
    cudaMemcpy(pvm->d_weight_mem, pvm->h_weight_mem,
               sizeof(float) * tw, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_dweight_mem[0], pvm->h_dweight_mem[0],
               sizeof(float) * tw, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_dweight_mem[1], pvm->h_dweight_mem[1],
               sizeof(float) * tw, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_weight_cache, pvm->h_weight_cache,
               sizeof(float) * tw, cudaMemcpyHostToDevice);

    for (int s = 0; s < PVM_SEQ_LENGTH; s++) {
        cudaMemcpy(pvm->d_input_activation[s], pvm->h_input_activation[s],
                   sizeof(float) * ti, cudaMemcpyHostToDevice);
        cudaMemcpy(pvm->d_input_delta[s], pvm->h_input_delta[s],
                   sizeof(float) * ti, cudaMemcpyHostToDevice);
        cudaMemcpy(pvm->d_input_error[s], pvm->h_input_error[s],
                   sizeof(float) * ti, cudaMemcpyHostToDevice);

        cudaMemcpy(pvm->d_output_activation[s], pvm->h_output_activation[s],
                   sizeof(float) * ti, cudaMemcpyHostToDevice);
        cudaMemcpy(pvm->d_output_delta[s], pvm->h_output_delta[s],
                   sizeof(float) * ti, cudaMemcpyHostToDevice);
        cudaMemcpy(pvm->d_output_error[s], pvm->h_output_error[s],
                   sizeof(float) * ti, cudaMemcpyHostToDevice);

        cudaMemcpy(pvm->d_repr_activation[s], pvm->h_repr_activation[s],
                   sizeof(float) * tr, cudaMemcpyHostToDevice);
        cudaMemcpy(pvm->d_repr_delta[s], pvm->h_repr_delta[s],
                   sizeof(float) * tr, cudaMemcpyHostToDevice);
        cudaMemcpy(pvm->d_repr_error[s], pvm->h_repr_error[s],
                   sizeof(float) * tr, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(pvm->d_beta_input, pvm->h_beta_input,
               sizeof(float) * ti, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_beta_repr, pvm->h_beta_repr,
               sizeof(float) * tr, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_learning_rate_arr, pvm->h_learning_rate_arr,
               sizeof(float) * pvm->total_units, cudaMemcpyHostToDevice);
    cudaMemcpy(pvm->d_momentum_arr, pvm->h_momentum_arr,
               sizeof(float) * pvm->total_units, cudaMemcpyHostToDevice);

    pvm->learning_rate = pvm->h_learning_rate_arr[0];

    /* Update CUDA device name for current machine */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    snprintf(pvm->device_name, sizeof(pvm->device_name), "%s", prop.name);

    printf("Loaded PVM model from %s (step=%d)\n", filename, pvm->step);
    return pvm;
}

/* =========================================================================
 *  Destroy
 * ========================================================================= */

void pvm_object_destroy(PVMObject *pvm) {
    if (!pvm) return;

    /* Free graph dynamic arrays */
    for (int i = 0; i < pvm->total_units; i++) {
        PVMBlock *b = &pvm->graph[i];
        free(b->primary_sources);
        free(b->context_sources);
        free(b->primary_destinations);
        free(b->context_destinations);
        free(b->xs);
        free(b->ys);
    }
    free(pvm->graph);
    free(pvm->layer_ptrs);

    /* Free host arrays */
    free(pvm->h_weight_mem);
    free(pvm->h_dweight_mem[0]);
    free(pvm->h_dweight_mem[1]);
    free(pvm->h_weight_cache);

    for (int s = 0; s < PVM_SEQ_LENGTH; s++) {
        free(pvm->h_input_activation[s]);
        free(pvm->h_input_delta[s]);
        free(pvm->h_input_error[s]);
        free(pvm->h_output_activation[s]);
        free(pvm->h_output_delta[s]);
        free(pvm->h_output_error[s]);
        free(pvm->h_repr_activation[s]);
        free(pvm->h_repr_delta[s]);
        free(pvm->h_repr_error[s]);
    }

    free(pvm->h_beta_input);
    free(pvm->h_beta_repr);
    free(pvm->h_learning_rate_arr);
    free(pvm->h_momentum_arr);

    free(pvm->h_weight_ptr0);
    free(pvm->h_weight_ptr1);
    free(pvm->h_input_ptr);
    free(pvm->h_repr_ptr);
    free(pvm->h_shape0_L0);
    free(pvm->h_shape1_L0);
    free(pvm->h_shape0_L1);
    free(pvm->h_shape1_L1);
    free(pvm->h_obj_id_L0);
    free(pvm->h_row_id_L0);
    free(pvm->h_obj_id_L1);
    free(pvm->h_row_id_L1);

    free(pvm->h_flow_from);
    free(pvm->h_flow_to);
    free(pvm->h_flow_size);
    free(pvm->h_flow_repr_from);
    free(pvm->h_flow_repr_to);
    free(pvm->h_flow_repr_size);
    free(pvm->h_flow_shift_from);
    free(pvm->h_flow_shift_to);
    free(pvm->h_flow_shift_size);
    free(pvm->h_flow_input_frame);
    free(pvm->h_flow_input_frame_size);

    free(pvm->h_learning_rate_saved);
    free(pvm->h_momentum_saved);

    /* Free device arrays */
    cudaFree(pvm->d_weight_mem);
    cudaFree(pvm->d_dweight_mem[0]);
    cudaFree(pvm->d_dweight_mem[1]);
    cudaFree(pvm->d_weight_cache);

    for (int s = 0; s < PVM_SEQ_LENGTH; s++) {
        cudaFree(pvm->d_input_activation[s]);
        cudaFree(pvm->d_input_delta[s]);
        cudaFree(pvm->d_input_error[s]);
        cudaFree(pvm->d_output_activation[s]);
        cudaFree(pvm->d_output_delta[s]);
        cudaFree(pvm->d_output_error[s]);
        cudaFree(pvm->d_repr_activation[s]);
        cudaFree(pvm->d_repr_delta[s]);
        cudaFree(pvm->d_repr_error[s]);
    }

    cudaFree(pvm->d_beta_input);
    cudaFree(pvm->d_beta_repr);
    cudaFree(pvm->d_learning_rate_arr);
    cudaFree(pvm->d_momentum_arr);

    cudaFree(pvm->d_weight_ptr0);
    cudaFree(pvm->d_weight_ptr1);
    cudaFree(pvm->d_input_ptr);
    cudaFree(pvm->d_repr_ptr);
    cudaFree(pvm->d_shape0_L0);
    cudaFree(pvm->d_shape1_L0);
    cudaFree(pvm->d_shape0_L1);
    cudaFree(pvm->d_shape1_L1);
    cudaFree(pvm->d_obj_id_L0);
    cudaFree(pvm->d_row_id_L0);
    cudaFree(pvm->d_obj_id_L1);
    cudaFree(pvm->d_row_id_L1);

    cudaFree(pvm->d_flow_from);
    cudaFree(pvm->d_flow_to);
    cudaFree(pvm->d_flow_size);
    cudaFree(pvm->d_flow_repr_from);
    cudaFree(pvm->d_flow_repr_to);
    cudaFree(pvm->d_flow_repr_size);
    cudaFree(pvm->d_flow_shift_from);
    cudaFree(pvm->d_flow_shift_to);
    cudaFree(pvm->d_flow_shift_size);
    cudaFree(pvm->d_flow_input_frame);
    cudaFree(pvm->d_flow_input_frame_size);

    free(pvm);
}
