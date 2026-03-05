/* pvm_graph.c – PVM graph construction (plain C99)
 * Direct port of sequence_learner.py generate_graph() /
 * generate_memory_ptrs() / generate_flow_ptrs()
 */
#include "pvm_graph.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

/* ── Dynamic int array helpers ─────────────────────────────────────────────── */
typedef struct { int *data; int n; int cap; } IntVec;

static void iv_init(IntVec *v)               { v->data=NULL; v->n=0; v->cap=0; }
static void iv_free(IntVec *v)               { free(v->data); iv_init(v); }

static void iv_push(IntVec *v, int val)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap*2 : 4;
        v->data = (int *)realloc(v->data, v->cap * sizeof(int));
    }
    v->data[v->n++] = val;
}

static int iv_contains(const IntVec *v, int val)
{
    for (int i = 0; i < v->n; ++i)
        if (v->data[i] == val) return 1;
    return 0;
}

static void iv_push_unique(IntVec *v, int val)
{
    if (!iv_contains(v, val)) iv_push(v, val);
}

/* Copy IntVec into a malloc'd int* (caller owns result) */
static int *iv_dup(const IntVec *v)
{
    if (!v->n) return NULL;
    int *d = (int *)malloc(v->n * sizeof(int));
    memcpy(d, v->data, v->n * sizeof(int));
    return d;
}

/* ── Connectivity helpers ──────────────────────────────────────────────────── */

/* Positions in a 2-D grid */
typedef struct { int x, y; } Pos;

/* Circular neighbourhood (taxicab approximation: circle by Euclidean radius) */
static void get_surround(Pos xy, int x_size, int y_size, int radius,
                          int exclude_self, IntVec *out_x, IntVec *out_y)
{
    iv_init(out_x); iv_init(out_y);
    for (int dx = -radius; dx <= radius; ++dx)
    for (int dy = -radius; dy <= radius; ++dy) {
        if (dx*dx + dy*dy > radius*radius) continue;
        int nx = xy.x + dx, ny = xy.y + dy;
        if (nx < 0 || nx >= x_size || ny < 0 || ny >= y_size) continue;
        if (exclude_self && dx == 0 && dy == 0) continue;
        iv_push(out_x, nx);
        iv_push(out_y, ny);
    }
}

/* Fan-in: units from lower layer that project onto upper-layer unit at xy */
static void get_fan_in(Pos xy,
                        int dim_x_l, int dim_y_l,
                        int dim_x_u, int dim_y_u,
                        int block_x, int block_y,
                        float radius,
                        IntVec *out_x, IntVec *out_y)
{
    iv_init(out_x); iv_init(out_y);

    float factor_x = (dim_x_u > 1)
        ? (float)((dim_x_l-1)-(block_x-1)) / (float)(dim_x_u-1)
        : (float)((dim_x_l-1)- block_x    ) / 2.0f;
    float factor_y = (dim_y_u > 1)
        ? (float)((dim_y_l-1)-(block_y-1)) / (float)(dim_y_u-1)
        : (float)((dim_y_l-1)- block_y    ) / 2.0f;

    float cx = (block_x - 1) * 0.5f;
    float cy = (block_y - 1) * 0.5f;

    float base_x, base_y;

    if (dim_x_u > 1 && dim_y_u > 1) {
        base_x = factor_x * xy.x;
        base_y = factor_y * xy.y;
    } else if (dim_x_u == 1 && dim_y_u > 1) {
        base_x = (dim_x_l - block_x) / 2.0f;
        base_y = factor_y * xy.y;
    } else if (dim_x_u > 1 && dim_y_u == 1) {
        base_x = factor_x * xy.x;
        base_y = (dim_y_l - block_y) / 2.0f;
    } else {
        base_x = (dim_x_l - block_x) / 2.0f;
        base_y = (dim_y_l - block_y) / 2.0f;
    }

    for (int xx = 0; xx < block_x; ++xx)
    for (int yy = 0; yy < block_y; ++yy) {
        float dx = xx - cx, dy = yy - cy;
        if (dx*dx + dy*dy > radius*radius) continue;
        iv_push(out_x, (int)(base_x + xx));
        iv_push(out_y, (int)(base_y + yy));
    }
}

/* ── Internal unit connectivity vecs (freed after build) ───────────────────── */
typedef struct {
    IntVec prim_src, prim_dst, ctx_src, ctx_dst;
} UnitConn;

/* ── Main build ────────────────────────────────────────────────────────────── */
void pvm_graph_build(PVMGraph *g, const PVMConfig *cfg)
{
    memset(g, 0, sizeof(*g));
    g->sequence_interval = 2;
    g->sequence_length   = 3;

    int seq_interval   = g->sequence_interval;
    int unit_size      = cfg->hidden_block_size * cfg->hidden_block_size;
    int patch_channels = cfg->input_channels;
    int patch_pixels   = cfg->input_block_size * cfg->input_block_size * patch_channels;
    int num_layers     = cfg->num_layers;

    /* Count total units */
    int total_units = 0;
    for (int l = 0; l < num_layers; ++l)
        total_units += cfg->layer_shapes[l] * cfg->layer_shapes[l];
    g->total_units = total_units;

    /* Allocate unit array */
    g->units = (PVMUnit *)calloc(total_units, sizeof(PVMUnit));

    /* Connectivity working arrays (temporary) */
    UnitConn *conn = (UnitConn *)calloc(total_units, sizeof(UnitConn));
    for (int i = 0; i < total_units; ++i) {
        iv_init(&conn[i].prim_src);
        iv_init(&conn[i].prim_dst);
        iv_init(&conn[i].ctx_src);
        iv_init(&conn[i].ctx_dst);
    }

    /* ── Phase 1: create units ─────────────────────────────────────────────── */
    int id = 0;
    for (int layer = 0; layer < num_layers; ++layer) {
        int L = cfg->layer_shapes[layer];
        g->layer_ptrs[layer] = (layer == 0) ? 0
            : g->layer_ptrs[layer-1] + cfg->layer_shapes[layer-1] * cfg->layer_shapes[layer-1];

        for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            PVMUnit *u   = &g->units[id];
            u->id        = id;
            u->layer     = layer;
            u->grid_x    = x;
            u->grid_y    = y;
            u->size      = unit_size;
            u->input_dim = 0;
            u->context_dim = 0;
            if (layer == 0) {
                u->base_input_offset  = seq_interval * patch_pixels;
                u->output_dim         = u->base_input_offset;
                u->base_input_size    = patch_pixels;
            } else {
                u->base_input_offset = 0;
                u->output_dim        = 0;
                u->base_input_size   = 0;
            }
            u->running_input_ptr = u->base_input_offset;
            ++id;
        }
    }

    /* ── Phase 2: lateral + primary connections ─────────────────────────────── */
    int total_primary = 0, total_context = 0;

    for (int ui = 0; ui < total_units; ++ui) {
        PVMUnit *u = &g->units[ui];
        int L = cfg->layer_shapes[u->layer];

        /* Lateral (surround) context connections */
        IntVec sx, sy;
        get_surround((Pos){u->grid_x, u->grid_y}, L, L,
                      cfg->lateral_radius, cfg->context_exclude_self, &sx, &sy);

        for (int vi = 0; vi < total_units; ++vi) {
            PVMUnit *v = &g->units[vi];
            if (v->layer != u->layer) continue;
            /* Is v in the surround of u? */
            int found = 0;
            for (int k = 0; k < sx.n; ++k)
                if (sx.data[k] == v->grid_x && sy.data[k] == v->grid_y) { found=1; break; }
            if (!found) continue;
            /* v sends context to u */
            iv_push_unique(&conn[vi].ctx_dst, ui);
            iv_push_unique(&conn[ui].ctx_src, vi);
            ++total_context;
        }
        iv_free(&sx); iv_free(&sy);

        /* Feed-forward (fan-in) from layer below */
        if (u->layer > 0) {
            int Ll = cfg->layer_shapes[u->layer - 1];
            IntVec fx, fy;
            get_fan_in((Pos){u->grid_x, u->grid_y},
                        Ll, Ll, L, L,
                        cfg->fan_in_square_size, cfg->fan_in_square_size,
                        cfg->fan_in_radius, &fx, &fy);

            for (int vi = 0; vi < total_units; ++vi) {
                PVMUnit *v = &g->units[vi];
                if (v->layer != u->layer - 1) continue;
                int found = 0;
                for (int k = 0; k < fx.n; ++k)
                    if (fx.data[k] == v->grid_x && fy.data[k] == v->grid_y) { found=1; break; }
                if (!found) continue;
                iv_push_unique(&conn[vi].prim_dst, ui);
                iv_push_unique(&conn[ui].prim_src, vi);
                u->base_input_size += g->units[vi].size;
                ++total_primary;
            }
            iv_free(&fx); iv_free(&fy);
        }
    }

    /* ── Phase 3: feedback connections ─────────────────────────────────────── */
    for (int ui = 0; ui < total_units; ++ui) {
        for (int di = 0; di < conn[ui].prim_dst.n; ++di) {
            int dest_id = conn[ui].prim_dst.data[di];
            iv_push_unique(&conn[dest_id].ctx_dst, ui);
            iv_push_unique(&conn[ui].ctx_src, dest_id);
            ++total_context;
        }
    }

    if (cfg->send_context_two_layers_back) {
        for (int ui = 0; ui < total_units; ++ui) {
            for (int di = 0; di < conn[ui].prim_dst.n; ++di) {
                int dest_id = conn[ui].prim_dst.data[di];
                for (int d2i = 0; d2i < conn[dest_id].prim_dst.n; ++d2i) {
                    int dest2_id = conn[dest_id].prim_dst.data[d2i];
                    iv_push_unique(&conn[dest2_id].ctx_dst, ui);
                    iv_push_unique(&conn[ui].ctx_src, dest2_id);
                    ++total_context;
                }
            }
        }
    }

    if (cfg->last_layer_context_to_all) {
        int last = num_layers - 1;
        for (int ui = 0; ui < total_units; ++ui) {
            PVMUnit *u = &g->units[ui];
            if (u->layer != last) continue;
            for (int vi = 0; vi < total_units; ++vi) {
                if (iv_contains(&conn[ui].ctx_dst, vi)) continue;
                iv_push(&conn[ui].ctx_dst, vi);
                iv_push(&conn[vi].ctx_src, ui);
                ++total_context;
            }
        }
    }
    g->total_primary_projections = total_primary;
    g->total_context_projections = total_context;

    /* ── Phase 4: compute input_dim / output_dim ───────────────────────────── */
    int total_weights = 0, total_input_m = 0, total_repr_m = 0;

    for (int ui = 0; ui < total_units; ++ui) {
        PVMUnit *u = &g->units[ui];
        u->input_dim    = u->base_input_offset;
        u->context_dim  = 0;

        for (int k = 0; k < conn[ui].prim_src.n; ++k) {
            int sid = conn[ui].prim_src.data[k];
            u->input_dim  += seq_interval * g->units[sid].size;
            u->output_dim += seq_interval * g->units[sid].size;
        }
        for (int k = 0; k < conn[ui].ctx_src.n; ++k) {
            int sid = conn[ui].ctx_src.data[k];
            u->input_dim   += g->units[sid].size;
            u->context_dim += g->units[sid].size;
        }

        if (cfg->feed_context_in_complex_layer) {
            total_weights += (u->input_dim + 1) * u->size
                           + (u->size + 1 + u->context_dim) * u->output_dim;
            total_repr_m  += u->size + 1 + u->context_dim;
        } else {
            total_weights += (u->input_dim + 1) * u->size
                           + (u->size + 1) * u->output_dim;
            total_repr_m  += u->size + 1;
        }
        total_input_m += u->input_dim + 1;
    }
    g->total_weights   = total_weights;
    g->total_input_mem = total_input_m;
    g->total_repr_mem  = total_repr_m;

    printf("Graph: %d units, %d weights\n", total_units, total_weights);
    printf("Input mem: %d  repr mem: %d\n", total_input_m, total_repr_m);
    printf("Primary proj: %d  context proj: %d\n", total_primary, total_context);

    /* Copy connectivity into per-unit arrays (freeze from work IntVecs) */
    for (int ui = 0; ui < total_units; ++ui) {
        PVMUnit *u = &g->units[ui];
        u->n_primary_sources      = conn[ui].prim_src.n;
        u->primary_sources        = iv_dup(&conn[ui].prim_src);
        u->n_context_sources      = conn[ui].ctx_src.n;
        u->context_sources        = iv_dup(&conn[ui].ctx_src);
        u->n_primary_destinations = conn[ui].prim_dst.n;
        u->primary_destinations   = iv_dup(&conn[ui].prim_dst);
        u->n_context_destinations = conn[ui].ctx_dst.n;
        u->context_destinations   = iv_dup(&conn[ui].ctx_dst);

        iv_free(&conn[ui].prim_src);
        iv_free(&conn[ui].prim_dst);
        iv_free(&conn[ui].ctx_src);
        iv_free(&conn[ui].ctx_dst);
    }
    free(conn);

    /* ── Phase 5: generate_memory_ptrs() ──────────────────────────────────── */
    g->weight_ptr0 = (int *)malloc(total_units * sizeof(int));
    g->weight_ptr1 = (int *)malloc(total_units * sizeof(int));
    g->shape0_L0   = (int *)malloc(total_units * sizeof(int));
    g->shape1_L0   = (int *)malloc(total_units * sizeof(int));
    g->shape0_L1   = (int *)malloc(total_units * sizeof(int));
    g->shape1_L1   = (int *)malloc(total_units * sizeof(int));
    g->input_ptr   = (int *)malloc(total_units * sizeof(int));
    g->repr_ptr    = (int *)malloc(total_units * sizeof(int));

    int curr_w0 = 0, curr_w1 = 0;
    int curr_i  = 0, curr_r  = 0;
    int thr_L0  = 0, thr_L1  = 0;

    for (int ui = 0; ui < total_units; ++ui) {
        PVMUnit *u = &g->units[ui];

        g->weight_ptr0[ui] = curr_w0;
        u->w0_ptr = curr_w0;
        curr_w0  += (u->input_dim + 1) * u->size;
        g->shape0_L0[ui] = u->size;
        g->shape1_L0[ui] = u->input_dim + 1;
        thr_L0 += u->input_dim + 1;

        g->weight_ptr1[ui] = curr_w1;
        u->w1_ptr = curr_w1;
        g->shape0_L1[ui] = u->output_dim;
        if (!cfg->feed_context_in_complex_layer) {
            g->shape1_L1[ui] = u->size + 1;
            curr_w1 += (u->size + 1) * u->output_dim;
            thr_L1  += u->size + 1;
        } else {
            g->shape1_L1[ui] = u->size + 1 + u->context_dim;
            curr_w1 += (u->size + 1 + u->context_dim) * u->output_dim;
            thr_L1  += u->size + 1 + u->context_dim;
        }

        g->input_ptr[ui] = curr_i;
        u->i_ptr = curr_i;
        curr_i  += u->input_dim + 1;

        g->repr_ptr[ui] = curr_r;
        u->r_ptr = curr_r;
        curr_r  += (cfg->feed_context_in_complex_layer)
                    ? u->size + 1 + u->context_dim
                    : u->size + 1;
    }

    /* Offset weight_ptr1 to sit after weight_ptr0 in the same flat weight array */
    for (int ui = 0; ui < total_units; ++ui) {
        g->weight_ptr1[ui] += curr_w0;
        g->units[ui].w1_ptr += curr_w0;
    }

    g->total_threads_L0 = thr_L0;
    g->total_threads_L1 = thr_L1;

    /* Build per-row dispatch arrays */
    g->obj_id_L0 = (int *)malloc(thr_L0 * sizeof(int));
    g->row_id_L0 = (int *)malloc(thr_L0 * sizeof(int));
    g->obj_id_L1 = (int *)malloc(thr_L1 * sizeof(int));
    g->row_id_L1 = (int *)malloc(thr_L1 * sizeof(int));

    int tid0 = 0, tid1 = 0;
    for (int ui = 0; ui < total_units; ++ui) {
        PVMUnit *u = &g->units[ui];
        for (int i = 0; i < u->input_dim + 1; ++i) {
            g->obj_id_L0[tid0] = ui;
            g->row_id_L0[tid0] = i;
            ++tid0;
        }
        int repr_dim = cfg->feed_context_in_complex_layer
                     ? u->size + 1 + u->context_dim : u->size + 1;
        for (int i = 0; i < repr_dim; ++i) {
            g->obj_id_L1[tid1] = ui;
            g->row_id_L1[tid1] = i;
            ++tid1;
        }
    }

    /* ── Phase 6: generate_flow_ptrs() ────────────────────────────────────── */
    int n_flow = total_primary + total_context;
    g->n_flow    = n_flow;
    g->flow_from = (int *)malloc(n_flow * sizeof(int));
    g->flow_to   = (int *)malloc(n_flow * sizeof(int));
    g->flow_size = (int *)malloc(n_flow * sizeof(int));

    g->n_repr_flow    = total_context;
    g->repr_flow_from = (int *)malloc(total_context * sizeof(int));
    g->repr_flow_to   = (int *)malloc(total_context * sizeof(int));
    g->repr_flow_size = (int *)malloc(total_context * sizeof(int));

    g->shift_from = (int *)malloc(total_units * sizeof(int));
    g->shift_to   = (int *)malloc(total_units * sizeof(int));
    g->shift_size = (int *)malloc(total_units * sizeof(int));

    int L0 = cfg->layer_shapes[0];
    g->n_frame_units    = L0 * L0;
    g->frame_ptr        = (int *)malloc(g->n_frame_units * sizeof(int));
    g->frame_ptr_size   = (int *)malloc(g->n_frame_units * sizeof(int));

    /* Reset running_input_ptr (it was set at unit creation) */
    for (int ui = 0; ui < total_units; ++ui)
        g->units[ui].running_input_ptr = g->units[ui].base_input_offset;

    int fi = 0;
    for (int ui = 0; ui < total_units; ++ui) {
        PVMUnit *u = &g->units[ui];
        int primary_src_shift = 0;

        for (int k = 0; k < u->n_primary_sources; ++k) {
            int sid = u->primary_sources[k];
            g->flow_size[fi] = unit_size;
            g->flow_from[fi] = g->repr_ptr[sid];
            g->flow_to  [fi] = u->i_ptr + u->running_input_ptr;
            primary_src_shift      += g->units[sid].size;
            u->running_input_ptr   += g->units[sid].size;
            ++fi;
        }
        u->running_input_ptr += primary_src_shift * (seq_interval - 1);

        for (int k = 0; k < u->n_context_sources; ++k) {
            int sid = u->context_sources[k];
            g->flow_size[fi] = unit_size;
            g->flow_from[fi] = g->repr_ptr[sid];
            g->flow_to  [fi] = u->i_ptr + u->running_input_ptr;
            u->running_input_ptr += g->units[sid].size;
            ++fi;
        }

        g->shift_from[ui] = u->i_ptr;
        g->shift_to  [ui] = u->i_ptr + u->base_input_size;
        g->shift_size[ui] = u->base_input_size * (seq_interval - 1);
    }

    int ri = 0;
    for (int ui = 0; ui < total_units; ++ui) {
        PVMUnit *u = &g->units[ui];
        int running = 0;
        for (int k = 0; k < u->n_context_sources; ++k) {
            int sid = u->context_sources[k];
            g->repr_flow_size[ri] = g->units[sid].size;
            g->repr_flow_from[ri] = g->repr_ptr[sid];
            g->repr_flow_to  [ri] = u->r_ptr + u->size + 1 + running;
            running += g->units[sid].size;
            ++ri;
        }
    }

    for (int i = 0; i < L0 * L0; ++i) {
        g->frame_ptr     [i] = g->units[i].i_ptr;
        g->frame_ptr_size[i] = g->units[i].base_input_size;
    }
}

/* ── Free all dynamic memory inside graph ─────────────────────────────────── */
void pvm_graph_free(PVMGraph *g)
{
    if (!g) return;
    for (int i = 0; i < g->total_units; ++i) {
        PVMUnit *u = &g->units[i];
        free(u->primary_sources);
        free(u->context_sources);
        free(u->primary_destinations);
        free(u->context_destinations);
    }
    free(g->units);
    free(g->weight_ptr0); free(g->weight_ptr1);
    free(g->shape0_L0);   free(g->shape1_L0);
    free(g->shape0_L1);   free(g->shape1_L1);
    free(g->input_ptr);   free(g->repr_ptr);
    free(g->obj_id_L0);   free(g->row_id_L0);
    free(g->obj_id_L1);   free(g->row_id_L1);
    free(g->flow_from);   free(g->flow_to);   free(g->flow_size);
    free(g->repr_flow_from); free(g->repr_flow_to); free(g->repr_flow_size);
    free(g->shift_from);  free(g->shift_to);  free(g->shift_size);
    free(g->frame_ptr);   free(g->frame_ptr_size);
    memset(g, 0, sizeof(*g));
}
