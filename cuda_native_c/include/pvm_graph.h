/* pvm_graph.h – PVM graph structure (plain C)
 * Mirrors sequence_learner.py generate_graph() logic.
 * All std::vector replaced with dynamically-allocated int* arrays + int count.
 */
#ifndef PVM_GRAPH_H
#define PVM_GRAPH_H

#include "pvm_config.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Per-unit connectivity (host side only, used during graph build) ───────── */
typedef struct {
    int id, layer;
    int grid_x, grid_y;
    int size;               /* hidden_block_size^2              */
    int input_dim;
    int output_dim;
    int context_dim;
    int base_input_size;
    int base_input_offset;

    /* connectivity lists – dynamic arrays */
    int *primary_sources;       int n_primary_sources;
    int *context_sources;       int n_context_sources;
    int *primary_destinations;  int n_primary_destinations;
    int *context_destinations;  int n_context_destinations;

    /* flat memory pointers into global arrays */
    int w0_ptr, w1_ptr;
    int i_ptr,  r_ptr;

    /* running pointer used during flow_ptr generation */
    int running_input_ptr;
} PVMUnit;

/* ── Graph (persistent, uploaded to GPU) ────────────────────────────────────── */
typedef struct {
    PVMUnit *units;
    int      total_units;

    /* layer start indices in units[] */
    int layer_ptrs[PVM_MAX_LAYERS];

    int total_weights;
    int total_input_mem;
    int total_repr_mem;
    int total_primary_projections;
    int total_context_projections;

    /* per-unit thread-dispatch arrays (length total_units) */
    int *weight_ptr0, *weight_ptr1;
    int *shape0_L0,   *shape1_L0;
    int *shape0_L1,   *shape1_L1;
    int *input_ptr,   *repr_ptr;

    /* dispatch index arrays (lengths below) */
    int *obj_id_L0, *row_id_L0; int total_threads_L0;
    int *obj_id_L1, *row_id_L1; int total_threads_L1;

    /* flow pointers – primary + context copies at each step */
    int *flow_from,   *flow_to,   *flow_size;
    int  n_flow;

    /* repr copy for feed_context_in_complex_layer */
    int *repr_flow_from, *repr_flow_to, *repr_flow_size;
    int  n_repr_flow;

    /* input temporal shift (one per unit) */
    int *shift_from, *shift_to, *shift_size;

    /* frame distribution (layer-0 units) */
    int *frame_ptr,  *frame_ptr_size;
    int  n_frame_units;

    int sequence_interval;  /* = 2 */
    int sequence_length;    /* = 3 */
} PVMGraph;

/* Build graph from config.  Allocates all arrays internally.
 * Call pvm_graph_free() when done. */
void pvm_graph_build(PVMGraph *g, const PVMConfig *cfg);

/* Free all dynamic memory inside graph (does not free g itself). */
void pvm_graph_free(PVMGraph *g);

#ifdef __cplusplus
}
#endif

#endif /* PVM_GRAPH_H */
