// PVM Graph Generation
// Direct C++ port of sequence_learner.py: generate_graph() + generate_memory_ptrs() + generate_flow_ptrs()

#include "pvm_graph.h"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

// ─── Connectivity helpers (mirror Python get_surround / get_fan_in) ───────────

using Pos = std::pair<int,int>;

static std::vector<Pos> get_surround(Pos xy, int x_size, int y_size,
                                      int radius, bool exclude_self)
{
    std::vector<Pos> res;
    for (int dx = -radius; dx <= radius; ++dx)
    for (int dy = -radius; dy <= radius; ++dy) {
        if (dx*dx + dy*dy > radius*radius) continue;
        int nx = xy.first  + dx;
        int ny = xy.second + dy;
        if (nx < 0 || nx >= x_size || ny < 0 || ny >= y_size) continue;
        if (exclude_self && dx == 0 && dy == 0) continue;
        res.push_back({nx, ny});
    }
    return res;
}

static std::vector<Pos> get_fan_in(Pos xy,
                                    int dim_x_l, int dim_y_l,
                                    int dim_x_u, int dim_y_u,
                                    int block_x, int block_y,
                                    float radius)
{
    int x = xy.first, y = xy.second;
    float factor_x = (dim_x_u > 1)
        ? (float)((dim_x_l-1)-(block_x-1)) / (dim_x_u-1)
        : (float)((dim_x_l-1)- block_x    ) / 2.0f;
    float factor_y = (dim_y_u > 1)
        ? (float)((dim_y_l-1)-(block_y-1)) / (dim_y_u-1)
        : (float)((dim_y_l-1)- block_y    ) / 2.0f;

    std::vector<Pos> res;
    float cx = (block_x-1)*0.5f, cy = (block_y-1)*0.5f;

    auto add = [&](int xx, int yy, float base_x, float base_y) {
        float dx = xx - cx, dy = yy - cy;
        if (dx*dx + dy*dy > radius*radius) return;
        res.push_back({(int)(base_x + xx), (int)(base_y + yy)});
    };

    if (dim_x_u > 1 && dim_y_u > 1) {
        for (int xx = 0; xx < block_x; ++xx)
        for (int yy = 0; yy < block_y; ++yy)
            add(xx, yy, factor_x*x, factor_y*y);
    } else if (dim_x_u == 1 && dim_y_u > 1) {
        float base_x = (dim_x_l - block_x) / 2.0f;
        for (int xx = 0; xx < block_x; ++xx)
        for (int yy = 0; yy < block_y; ++yy)
            add(xx, yy, base_x, factor_y*y);
    } else if (dim_x_u > 1 && dim_y_u == 1) {
        float base_y = (dim_y_l - block_y) / 2.0f;
        for (int xx = 0; xx < block_x; ++xx)
        for (int yy = 0; yy < block_y; ++yy)
            add(xx, yy, factor_x*x, base_y);
    } else {
        float base_x = (dim_x_l - block_x) / 2.0f;
        float base_y = (dim_y_l - block_y) / 2.0f;
        for (int xx = 0; xx < block_x; ++xx)
        for (int yy = 0; yy < block_y; ++yy)
            add(xx, yy, base_x, base_y);
    }
    return res;
}

static void append_unique(std::vector<int>& v, int el) {
    if (std::find(v.begin(), v.end(), el) == v.end()) v.push_back(el);
}

// ─── Main Build ──────────────────────────────────────────────────────────────

PVMGraph build_graph(const PVMConfig& cfg)
{
    PVMGraph G;
    G.sequence_interval = 2;
    G.sequence_length   = 3;

    int seq_interval   = G.sequence_interval;
    int unit_size      = cfg.hidden_block_size * cfg.hidden_block_size;
    int patch_channels = cfg.input_channels;
    int patch_pixels   = cfg.input_block_size * cfg.input_block_size * patch_channels;

    int num_layers = cfg.num_layers();

    // ── Phase 1: create units ────────────────────────────────────────────────
    int id = 0;
    for (int layer = 0; layer < num_layers; ++layer) {
        int L = cfg.layer_shapes[layer];

        // accumulate layer_ptr
        if (layer == 0) G.layer_ptrs.push_back(0);
        else G.layer_ptrs.push_back(G.layer_ptrs.back() + cfg.layer_shapes[layer-1] * cfg.layer_shapes[layer-1]);

        for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            PVMUnit u;
            u.id     = id++;
            u.layer  = layer;
            u.grid_x = x;
            u.grid_y = y;
            u.size   = unit_size;
            u.input_dim  = 0;
            u.context_dim = 0;

            if (layer == 0) {
                u.base_input_offset = seq_interval * patch_pixels;
                u.output_dim   = u.base_input_offset;
                u.base_input_size = patch_pixels;
            } else {
                u.base_input_offset = 0;
                u.output_dim   = 0;
                u.base_input_size = 0;
            }
            u.running_input_ptr = u.base_input_offset;
            G.units.push_back(u);
        }
    }
    G.total_units = (int)G.units.size();

    // ── Phase 2: lateral + primary connections ───────────────────────────────
    int total_primary  = 0;
    int total_context  = 0;

    for (auto& u : G.units) {
        int L = cfg.layer_shapes[u.layer];

        // lateral (surround) connections
        auto sur = get_surround({u.grid_x, u.grid_y}, L, L,
                                 cfg.lateral_radius, cfg.context_exclude_self);

        for (auto& v : G.units) {
            if (v.layer != u.layer) continue;
            Pos vp{v.grid_x, v.grid_y};
            if (std::find(sur.begin(), sur.end(), vp) == sur.end()) continue;
            // v is in surround of u => v sends context to u
            append_unique(v.context_destinations, u.id);
            append_unique(u.context_sources, v.id);
            ++total_context;
        }

        // feed-forward (fan-in) connections from layer below
        if (u.layer > 0) {
            int Ll = cfg.layer_shapes[u.layer - 1];
            auto fan = get_fan_in({u.grid_x, u.grid_y},
                                   Ll, Ll,
                                   L, L,
                                   cfg.fan_in_square_size, cfg.fan_in_square_size,
                                   cfg.fan_in_radius);

            // count layer below start id
            int below_start = G.layer_ptrs[u.layer - 1];
            for (auto& v : G.units) {
                if (v.layer != u.layer - 1) continue;
                Pos vp{v.grid_x, v.grid_y};
                if (std::find(fan.begin(), fan.end(), vp) == fan.end()) continue;

                append_unique(v.primary_destinations, u.id);
                append_unique(u.primary_sources, v.id);
                u.base_input_size += v.size;
                ++total_primary;
            }
        }
    }

    // ── Phase 3: feedback connections ────────────────────────────────────────
    for (auto& u : G.units) {
        for (int dest_id : u.primary_destinations) {
            auto& dest = G.units[dest_id];
            append_unique(dest.context_destinations, u.id);
            append_unique(u.context_sources, dest_id);
            ++total_context;
        }
    }

    if (cfg.send_context_two_layers_back) {
        for (auto& u : G.units) {
            for (int dest_id : u.primary_destinations) {
                for (int dest2_id : G.units[dest_id].primary_destinations) {
                    auto& dest2 = G.units[dest2_id];
                    append_unique(dest2.context_destinations, u.id);
                    append_unique(u.context_sources, dest2_id);
                    ++total_context;
                }
            }
        }
    }

    if (cfg.last_layer_context_to_all) {
        int last_layer = num_layers - 1;
        for (auto& u : G.units) {
            if (u.layer != last_layer) continue;
            for (auto& v : G.units) {
                if (std::find(u.context_destinations.begin(),
                              u.context_destinations.end(), v.id) != u.context_destinations.end()) continue;
                u.context_destinations.push_back(v.id);
                v.context_sources.push_back(u.id);
                ++total_context;
            }
        }
    }

    G.total_primary_projections  = total_primary;
    G.total_context_projections  = total_context;

    // ── Phase 4: compute input_dim / output_dim per unit ────────────────────
    int total_weights  = 0;
    int total_input_m  = 0;
    int total_repr_m   = 0;

    for (auto& u : G.units) {
        u.input_dim = u.base_input_offset;
        for (int sid : u.primary_sources) {
            u.input_dim  += seq_interval * G.units[sid].size;
            u.output_dim += seq_interval * G.units[sid].size;
        }
        for (int sid : u.context_sources) {
            u.input_dim  += G.units[sid].size;
            u.context_dim += G.units[sid].size;
        }

        if (cfg.feed_context_in_complex_layer) {
            total_weights += (u.input_dim + 1) * u.size +
                             (u.size + 1 + u.context_dim) * u.output_dim;
            total_repr_m  += u.size + 1 + u.context_dim;
        } else {
            total_weights += (u.input_dim + 1) * u.size +
                             (u.size + 1) * u.output_dim;
            total_repr_m  += u.size + 1;
        }
        total_input_m += u.input_dim + 1;
    }
    G.total_weights   = total_weights;
    G.total_input_mem = total_input_m;
    G.total_repr_mem  = total_repr_m;

    printf("Graph: %d units, %d weights\n", G.total_units, G.total_weights);
    printf("Input mem: %d, repr mem: %d\n", G.total_input_mem, G.total_repr_mem);
    printf("Primary proj: %d, context proj: %d\n", total_primary, total_context);

    // ── Phase 5: generate_memory_ptrs() ─────────────────────────────────────
    G.weight_ptr0.resize(G.total_units);
    G.weight_ptr1.resize(G.total_units);
    G.shape0_L0.resize(G.total_units);
    G.shape1_L0.resize(G.total_units);
    G.shape0_L1.resize(G.total_units);
    G.shape1_L1.resize(G.total_units);
    G.input_ptr.resize(G.total_units);
    G.repr_ptr.resize(G.total_units);

    int curr_w0 = 0, curr_w1 = 0;
    int curr_i  = 0, curr_r  = 0;
    int total_threads_L0 = 0, total_threads_L1 = 0;

    for (auto& u : G.units) {
        G.weight_ptr0[u.id] = curr_w0;
        u.w0_ptr = curr_w0;
        curr_w0 += (u.input_dim + 1) * u.size;
        G.shape0_L0[u.id] = u.size;
        G.shape1_L0[u.id] = u.input_dim + 1;
        total_threads_L0  += u.input_dim + 1;

        G.weight_ptr1[u.id] = curr_w1;
        u.w1_ptr = curr_w1;
        G.shape0_L1[u.id] = u.output_dim;
        if (!cfg.feed_context_in_complex_layer) {
            G.shape1_L1[u.id] = u.size + 1;
            curr_w1 += (u.size + 1) * u.output_dim;
            total_threads_L1 += u.size + 1;
        } else {
            G.shape1_L1[u.id] = u.size + 1 + u.context_dim;
            curr_w1 += (u.size + 1 + u.context_dim) * u.output_dim;
            total_threads_L1 += u.size + 1 + u.context_dim;
        }

        G.input_ptr[u.id] = curr_i;
        u.i_ptr = curr_i;
        curr_i += u.input_dim + 1;

        G.repr_ptr[u.id] = curr_r;
        u.r_ptr = curr_r;
        if (!cfg.feed_context_in_complex_layer)
            curr_r += u.size + 1;
        else
            curr_r += u.size + 1 + u.context_dim;
    }

    // Offset w1 to sit after w0 in the same flat array
    for (int i = 0; i < G.total_units; ++i)
        G.weight_ptr1[i] += curr_w0;
    for (auto& u : G.units)
        u.w1_ptr += curr_w0;

    // Build per-row dispatch arrays
    G.obj_id_L0.resize(total_threads_L0);
    G.row_id_L0.resize(total_threads_L0);
    G.obj_id_L1.resize(total_threads_L1);
    G.row_id_L1.resize(total_threads_L1);

    int tid_L0 = 0, tid_L1 = 0;
    for (auto& u : G.units) {
        for (int i = 0; i < u.input_dim + 1; ++i) {
            G.obj_id_L0[tid_L0] = u.id;
            G.row_id_L0[tid_L0] = i;
            ++tid_L0;
        }
        int repr_dim = cfg.feed_context_in_complex_layer
                     ? u.size + 1 + u.context_dim : u.size + 1;
        for (int i = 0; i < repr_dim; ++i) {
            G.obj_id_L1[tid_L1] = u.id;
            G.row_id_L1[tid_L1] = i;
            ++tid_L1;
        }
    }
    G.total_threads_L0 = total_threads_L0;
    G.total_threads_L1 = total_threads_L1;

    // ── Phase 6: generate_flow_ptrs() ───────────────────────────────────────
    int n_flow = total_primary + total_context;
    G.flow_ptr_from.resize(n_flow);
    G.flow_ptr_to.resize(n_flow);
    G.flow_ptr_size.resize(n_flow);
    G.flow_ptr_repr_from.resize(total_context);
    G.flow_ptr_repr_to.resize(total_context);
    G.flow_ptr_repr_size.resize(total_context);
    G.flow_ptr_input_shift_from.resize(G.total_units);
    G.flow_ptr_input_shift_to.resize(G.total_units);
    G.flow_ptr_input_shift_size.resize(G.total_units);

    int L0 = cfg.layer_shapes[0];
    G.flow_ptr_input_frame.resize(L0 * L0);
    G.flow_ptr_input_frame_size.resize(L0 * L0);

    int fi = 0;
    for (auto& u : G.units) {
        int primary_src_shift = 0;
        for (int sid : u.primary_sources) {
            G.flow_ptr_size[fi] = unit_size;
            G.flow_ptr_from[fi] = G.repr_ptr[sid];
            G.flow_ptr_to[fi]   = u.i_ptr + u.running_input_ptr;
            primary_src_shift  += G.units[sid].size;
            u.running_input_ptr += G.units[sid].size;
            ++fi;
        }
        u.running_input_ptr += primary_src_shift * (seq_interval - 1);
        for (int sid : u.context_sources) {
            G.flow_ptr_size[fi] = unit_size;
            G.flow_ptr_from[fi] = G.repr_ptr[sid];
            G.flow_ptr_to[fi]   = u.i_ptr + u.running_input_ptr;
            u.running_input_ptr += G.units[sid].size;
            ++fi;
        }
        G.flow_ptr_input_shift_from[u.id] = u.i_ptr;
        G.flow_ptr_input_shift_to[u.id]   = u.i_ptr + u.base_input_size;
        G.flow_ptr_input_shift_size[u.id] = u.base_input_size * (seq_interval - 1);
    }

    int ri = 0;
    for (auto& u : G.units) {
        int running = 0;
        for (int sid : u.context_sources) {
            G.flow_ptr_repr_size[ri] = G.units[sid].size;
            G.flow_ptr_repr_from[ri] = G.repr_ptr[sid];
            G.flow_ptr_repr_to[ri]   = u.r_ptr + u.size + 1 + running;
            running += G.units[sid].size;
            ++ri;
        }
    }

    for (int i = 0; i < L0 * L0; ++i) {
        G.flow_ptr_input_frame[i]      = G.units[i].i_ptr;
        G.flow_ptr_input_frame_size[i] = G.units[i].base_input_size;
    }

    return G;
}
