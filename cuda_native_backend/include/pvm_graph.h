#pragma once
// PVM Graph Generation - mirrors Python sequence_learner.py generate_graph()
// Computes the full connectivity between PVM units

#include "pvm_config.h"
#include <vector>
#include <utility>

struct PVMUnit {
    int id;
    int layer;
    int grid_x, grid_y;

    // dimensions
    int size;            // hidden_block_size^2
    int input_dim;       // total input vector size
    int output_dim;      // total output vector size (= prediction target)
    int context_dim;
    int base_input_size;   // pixel patch size (layer 0 only)
    int base_input_offset; // pixel patch offset in input buffer (layer 0: seq_interval * patch)

    // connectivity
    std::vector<int> primary_sources;
    std::vector<int> context_sources;
    std::vector<int> primary_destinations;
    std::vector<int> context_destinations;

    // flat memory pointers (indices into global arrays)
    int w0_ptr;   // weight layer 0 start
    int w1_ptr;   // weight layer 1 start
    int i_ptr;    // input mem start
    int r_ptr;    // repr mem start

    // running pointer used during flow_ptr generation
    int running_input_ptr;
};

struct PVMGraph {
    std::vector<PVMUnit> units;
    std::vector<int>     layer_ptrs;  // index where each layer starts in units array

    int total_units              = 0;
    int total_weights            = 0;
    int total_input_mem          = 0;
    int total_repr_mem           = 0;
    int total_primary_projections   = 0;
    int total_context_projections   = 0;

    // Flat thread-dispatch index arrays (obj_id / row_id)
    std::vector<int> obj_id_L0, row_id_L0;  // for forward L0 kernel
    std::vector<int> obj_id_L1, row_id_L1;  // for forward L1 kernel

    // Per-unit shapes (needed on GPU)
    std::vector<int> weight_ptr0, weight_ptr1;
    std::vector<int> shape0_L0, shape1_L0;
    std::vector<int> shape0_L1, shape1_L1;
    std::vector<int> input_ptr, repr_ptr;

    // Flow pointers – primary + context copy at each step
    std::vector<int> flow_ptr_from, flow_ptr_to, flow_ptr_size;

    // Repr copy for feed_context_in_complex_layer
    std::vector<int> flow_ptr_repr_from, flow_ptr_repr_to, flow_ptr_repr_size;

    // Input shift (temporal shift of pixel blocks)
    std::vector<int> flow_ptr_input_shift_from;
    std::vector<int> flow_ptr_input_shift_to;
    std::vector<int> flow_ptr_input_shift_size;

    // Frame distribution (layer 0 units)
    std::vector<int> flow_ptr_input_frame;
    std::vector<int> flow_ptr_input_frame_size;

    int sequence_interval = 2;
    int sequence_length   = 3;

    int total_threads_L0 = 0;
    int total_threads_L1 = 0;
};

// Build the full graph from a config
PVMGraph build_graph(const PVMConfig& cfg);
