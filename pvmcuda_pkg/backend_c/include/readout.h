#pragma once
#include "pvm_model.h"
#include "mlp.h"

typedef struct {
    PVMObject *pvm;  // reference, NOT owned
    MLPCollection *mlp;
    int readout_layer;  // typically 2 (or 3 for 4_layer_readout)
    int heatmap_block_size;
    int blocks_x, blocks_y;
    int shape;  // blocks_x * heatmap_block_size
    int total_units;  // blocks_x * blocks_y
    int total_blocks;  // number of copy entries
    int opt_abs_diff;

    // Host arrays
    int *h_ptrs_from;
    int *h_ptrs_to;
    int *h_qnt_from;
    int *h_sizes;  // per-readout-unit total input size

    // Device arrays
    int *d_ptrs_from;
    int *d_ptrs_to;
    int *d_qnt_from;
} ReadoutObject;

ReadoutObject* readout_create(PVMObject *pvm, int representation_size, int heatmap_block_size);
void readout_destroy(ReadoutObject *ro);
void readout_copy_data(ReadoutObject *ro);
void readout_forward(ReadoutObject *ro);
void readout_train(ReadoutObject *ro, const float *label, int h, int w);
void readout_get_heatmap(ReadoutObject *ro, float *out_buf);
void readout_update_learning_rate(ReadoutObject *ro, float override_rate);
void readout_set_pvm(ReadoutObject *ro, PVMObject *pvm);
int readout_save(ReadoutObject *ro, const char *filename);
ReadoutObject* readout_load(const char *filename);
