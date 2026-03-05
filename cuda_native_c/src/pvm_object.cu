/* pvm_object.cu – PVM training object (CUDA C implementation)
 * Compiled as CUDA C (nvcc -x cu).  Uses:
 *   - CUDA Streams: frame upload overlaps compute
 *   - CUDA Graphs:  forward+backward captured once, replayed at near-zero CPU cost
 *   - cuBLAS:       saxpy / sscal for error computation
 */
#include "pvm_object.h"
#include "pvm_kernels.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ── Handle for cuBLAS (one global per process) ─────────────────────────── */
static cublasHandle_t s_cublas = NULL;
static int            s_cublas_refs = 0;

static void cublas_acquire(cudaStream_t stream)
{
    if (!s_cublas) {
        cublasCreate(&s_cublas);
    }
    cublasSetStream(s_cublas, stream);
    s_cublas_refs++;
}
static void cublas_release(void)
{
    if (--s_cublas_refs == 0) {
        cublasDestroy(s_cublas);
        s_cublas = NULL;
    }
}

/* ── CUDA / cuBLAS error-check macros ───────────────────────────────────── */
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error '%s' at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _e = (x); \
    if (_e != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/* ── GPU buffer helpers ──────────────────────────────────────────────────── */
static void gpu_buf_alloc_f(GpuBuf *b, size_t n_floats)
{
    b->bytes = n_floats * sizeof(float);
    CUDA_CHECK(cudaMalloc(&b->ptr, b->bytes));
    CUDA_CHECK(cudaMemset(b->ptr, 0, b->bytes));
}

static void gpu_buf_free_f(GpuBuf *b)
{
    if (b->ptr) { cudaFree(b->ptr); b->ptr = NULL; b->bytes = 0; }
}

static void gpu_intbuf_alloc(GpuIntBuf *b, const int *data, int n)
{
    b->bytes = (size_t)n * sizeof(int);
    CUDA_CHECK(cudaMalloc(&b->ptr, b->bytes));
    CUDA_CHECK(cudaMemcpy(b->ptr, data, b->bytes, cudaMemcpyHostToDevice));
}

static void gpu_intbuf_free(GpuIntBuf *b)
{
    if (b->ptr) { cudaFree(b->ptr); b->ptr = NULL; b->bytes = 0; }
}

/* ── Slot helpers ─────────────────────────────────────────────────────────── */
static inline int cur_slot(const PVMObject *o)
{
    return (int)(o->step % o->graph.sequence_length);
}
static inline int prev_slot(const PVMObject *o)
{
    return (int)((o->step - 1 + o->graph.sequence_length) % o->graph.sequence_length);
}
static inline int lagged_slot(const PVMObject *o)
{
    return (int)((o->step - o->graph.sequence_interval + o->graph.sequence_length)
                 % o->graph.sequence_length);
}

/* ── StepState allocation ────────────────────────────────────────────────── */
static void step_state_alloc(StepState *s, int batch, int im, int rm)
{
    size_t in_sz   = (size_t)batch * im;
    size_t repr_sz = (size_t)batch * rm;
    gpu_buf_alloc_f(&s->input_activ,   in_sz);
    gpu_buf_alloc_f(&s->input_error,   in_sz);
    gpu_buf_alloc_f(&s->input_delta,   in_sz);
    gpu_buf_alloc_f(&s->output_activ,  in_sz);
    gpu_buf_alloc_f(&s->output_error,  in_sz);
    gpu_buf_alloc_f(&s->output_delta,  in_sz);
    gpu_buf_alloc_f(&s->repr_activ,    repr_sz);
    gpu_buf_alloc_f(&s->repr_error,    repr_sz);
    gpu_buf_alloc_f(&s->repr_delta,    repr_sz);
}

static void step_state_free(StepState *s)
{
    gpu_buf_free_f(&s->input_activ);  gpu_buf_free_f(&s->input_error);
    gpu_buf_free_f(&s->input_delta);
    gpu_buf_free_f(&s->output_activ); gpu_buf_free_f(&s->output_error);
    gpu_buf_free_f(&s->output_delta);
    gpu_buf_free_f(&s->repr_activ);   gpu_buf_free_f(&s->repr_error);
    gpu_buf_free_f(&s->repr_delta);
}

/* ── pvm_object_create ───────────────────────────────────────────────────── */
PVMObject *pvm_object_create(const PVMConfig *cfg)
{
    PVMObject *obj = (PVMObject *)calloc(1, sizeof(PVMObject));
    obj->cfg       = *cfg;
    obj->batch_size = cfg->batch_size;
    obj->step       = 0;
    obj->buf_idx    = 0;
    obj->graph_captured = 0;

    /* Streams & events */
    CUDA_CHECK(cudaStreamCreate(&obj->stream_compute));
    CUDA_CHECK(cudaStreamCreate(&obj->stream_upload));
    CUDA_CHECK(cudaEventCreate(&obj->event_upload_done));

    /* cuBLAS */
    cublas_acquire(obj->stream_compute);

    /* Build graph */
    pvm_graph_build(&obj->graph, cfg);

    int  W  = obj->graph.total_weights;
    int  IM = obj->graph.total_input_mem;
    int  RM = obj->graph.total_repr_mem;
    int  U  = obj->graph.total_units;
    int  B  = obj->batch_size;
    int  SL = obj->graph.sequence_length;

    /* Weights */
    gpu_buf_alloc_f(&obj->weight_main,  (size_t)W);
    gpu_buf_alloc_f(&obj->dweight[0],   (size_t)W);
    gpu_buf_alloc_f(&obj->dweight[1],   (size_t)W);
    gpu_buf_alloc_f(&obj->weight_cache, (size_t)W);

    /* Initialise weights: uniform(-0.015, 0.015) via a simple LCG */
    {
        float *wh = (float *)malloc((size_t)W * sizeof(float));
        uint64_t rng = 6364136223846793005ULL;
        for (int i = 0; i < W; ++i) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            float u = (float)((rng >> 33) & 0x7FFFFFFF) / (float)0x80000000;
            wh[i] = (u - 0.5f) * 0.03f;
        }
        CUDA_CHECK(cudaMemcpy(obj->weight_main.ptr, wh, (size_t)W * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(wh);
    }

    /* Per-timestep states */
    for (int i = 0; i < SL && i < PVM_MAX_SEQ; ++i)
        step_state_alloc(&obj->states[i], B, IM, RM);

    /* Beta arrays (all ones) */
    {
        gpu_buf_alloc_f(&obj->beta_input, (size_t)IM);
        gpu_buf_alloc_f(&obj->beta_repr,  (size_t)RM);
        float *tmp = (float *)malloc((size_t)(IM > RM ? IM : RM) * sizeof(float));
        for (int i = 0; i < IM; ++i) tmp[i] = 1.f;
        CUDA_CHECK(cudaMemcpy(obj->beta_input.ptr, tmp, (size_t)IM*sizeof(float),
                              cudaMemcpyHostToDevice));
        for (int i = 0; i < RM; ++i) tmp[i] = 1.f;
        CUDA_CHECK(cudaMemcpy(obj->beta_repr.ptr, tmp, (size_t)RM*sizeof(float),
                              cudaMemcpyHostToDevice));
        free(tmp);
    }

    /* LR / momentum arrays (start zeroed) */
    gpu_buf_alloc_f(&obj->lr_arr,       (size_t)U);
    gpu_buf_alloc_f(&obj->momentum_arr, (size_t)U);

    /* GPU frame buffer */
    int H = pvm_config_input_size(cfg);
    gpu_buf_alloc_f(&obj->gpu_frame, (size_t)B * H * H * cfg->input_channels);

    /* Pinned host staging buffer */
    obj->pinned_frame_bytes = (size_t)B * H * H * cfg->input_channels * sizeof(float);
    CUDA_CHECK(cudaMallocHost(&obj->pinned_frame, obj->pinned_frame_bytes));

    /* Upload graph metadata to GPU */
    PVMGraph *g = &obj->graph;
    gpu_intbuf_alloc(&obj->g_weight_ptr0, g->weight_ptr0, U);
    gpu_intbuf_alloc(&obj->g_weight_ptr1, g->weight_ptr1, U);
    gpu_intbuf_alloc(&obj->g_shape0_L0,   g->shape0_L0,   U);
    gpu_intbuf_alloc(&obj->g_shape1_L0,   g->shape1_L0,   U);
    gpu_intbuf_alloc(&obj->g_shape0_L1,   g->shape0_L1,   U);
    gpu_intbuf_alloc(&obj->g_shape1_L1,   g->shape1_L1,   U);
    gpu_intbuf_alloc(&obj->g_input_ptr,   g->input_ptr,   U);
    gpu_intbuf_alloc(&obj->g_repr_ptr,    g->repr_ptr,    U);
    gpu_intbuf_alloc(&obj->g_obj_id_L0,   g->obj_id_L0,   g->total_threads_L0);
    gpu_intbuf_alloc(&obj->g_row_id_L0,   g->row_id_L0,   g->total_threads_L0);
    gpu_intbuf_alloc(&obj->g_obj_id_L1,   g->obj_id_L1,   g->total_threads_L1);
    gpu_intbuf_alloc(&obj->g_row_id_L1,   g->row_id_L1,   g->total_threads_L1);

    gpu_intbuf_alloc(&obj->g_flow_from,   g->flow_from,   g->n_flow);
    gpu_intbuf_alloc(&obj->g_flow_to,     g->flow_to,     g->n_flow);
    gpu_intbuf_alloc(&obj->g_flow_size,   g->flow_size,   g->n_flow);

    gpu_intbuf_alloc(&obj->g_repr_flow_from, g->repr_flow_from, g->n_repr_flow);
    gpu_intbuf_alloc(&obj->g_repr_flow_to,   g->repr_flow_to,   g->n_repr_flow);
    gpu_intbuf_alloc(&obj->g_repr_flow_size, g->repr_flow_size, g->n_repr_flow);

    gpu_intbuf_alloc(&obj->g_shift_from, g->shift_from, U);
    gpu_intbuf_alloc(&obj->g_shift_to,   g->shift_to,   U);
    gpu_intbuf_alloc(&obj->g_shift_size, g->shift_size, U);

    gpu_intbuf_alloc(&obj->g_frame_ptr,      g->frame_ptr,      g->n_frame_units);
    gpu_intbuf_alloc(&obj->g_frame_ptr_size, g->frame_ptr_size, g->n_frame_units);

    return obj;
}

/* ── pvm_object_destroy ──────────────────────────────────────────────────── */
void pvm_object_destroy(PVMObject *obj)
{
    if (!obj) return;

    CUDA_CHECK(cudaStreamSynchronize(obj->stream_compute));

    if (obj->graph_captured) {
        cudaGraphExecDestroy(obj->cuda_graph_exec);
        cudaGraphDestroy(obj->cuda_graph);
    }

    for (int i = 0; i < obj->graph.sequence_length && i < PVM_MAX_SEQ; ++i)
        step_state_free(&obj->states[i]);

    gpu_buf_free_f(&obj->weight_main);
    gpu_buf_free_f(&obj->dweight[0]);
    gpu_buf_free_f(&obj->dweight[1]);
    gpu_buf_free_f(&obj->weight_cache);
    gpu_buf_free_f(&obj->beta_input);
    gpu_buf_free_f(&obj->beta_repr);
    gpu_buf_free_f(&obj->lr_arr);
    gpu_buf_free_f(&obj->momentum_arr);
    gpu_buf_free_f(&obj->gpu_frame);

    gpu_intbuf_free(&obj->g_weight_ptr0); gpu_intbuf_free(&obj->g_weight_ptr1);
    gpu_intbuf_free(&obj->g_shape0_L0);   gpu_intbuf_free(&obj->g_shape1_L0);
    gpu_intbuf_free(&obj->g_shape0_L1);   gpu_intbuf_free(&obj->g_shape1_L1);
    gpu_intbuf_free(&obj->g_input_ptr);   gpu_intbuf_free(&obj->g_repr_ptr);
    gpu_intbuf_free(&obj->g_obj_id_L0);   gpu_intbuf_free(&obj->g_row_id_L0);
    gpu_intbuf_free(&obj->g_obj_id_L1);   gpu_intbuf_free(&obj->g_row_id_L1);
    gpu_intbuf_free(&obj->g_flow_from);   gpu_intbuf_free(&obj->g_flow_to);
    gpu_intbuf_free(&obj->g_flow_size);
    gpu_intbuf_free(&obj->g_repr_flow_from); gpu_intbuf_free(&obj->g_repr_flow_to);
    gpu_intbuf_free(&obj->g_repr_flow_size);
    gpu_intbuf_free(&obj->g_shift_from);  gpu_intbuf_free(&obj->g_shift_to);
    gpu_intbuf_free(&obj->g_shift_size);
    gpu_intbuf_free(&obj->g_frame_ptr);   gpu_intbuf_free(&obj->g_frame_ptr_size);

    if (obj->pinned_frame) cudaFreeHost(obj->pinned_frame);
    cudaEventDestroy(obj->event_upload_done);
    cudaStreamDestroy(obj->stream_upload);
    cudaStreamDestroy(obj->stream_compute);
    cublas_release();

    pvm_graph_free(&obj->graph);
    free(obj);
}

/* ── pvm_push_input ──────────────────────────────────────────────────────── */
/* Mirrors Python push_input_gpu():
 *   1. Upload frame to GPU (upload stream, event-based sync)
 *   2. Temporal shift of pixel blocks in input memory
 *   3. Copy repr->input (primary + context, with compression)
 *   4. Distribute frame patches to layer-0 units                           */
void pvm_push_input(PVMObject *obj, const float *frame_host)
{
    cudaStream_t stream = obj->stream_compute;
    PVMGraph *g    = &obj->graph;
    int U          = g->total_units;
    int B          = obj->batch_size;
    int cur        = cur_slot(obj);
    int prev       = prev_slot(obj);

    /* Upload frame asynchronously on the upload stream */
    CUDA_CHECK(cudaMemcpyAsync(obj->gpu_frame.ptr, frame_host,
                               obj->pinned_frame_bytes,
                               cudaMemcpyHostToDevice, obj->stream_upload));
    CUDA_CHECK(cudaEventRecord(obj->event_upload_done, obj->stream_upload));

    /* 1. Temporal shift: copy pixel block one step forward in time */
    {
        int blk = 128, grd = (U + blk - 1) / blk;
        k_copy_blocks<<<grd, blk, 0, stream>>>(
            obj->states[prev].input_activ.ptr,
            obj->states[cur ].input_activ.ptr,
            obj->g_shift_from.ptr,
            obj->g_shift_size.ptr,
            obj->g_shift_to.ptr,
            U);
    }

    /* Zero repr for current step */
    CUDA_CHECK(cudaMemsetAsync(obj->states[cur].repr_activ.ptr, 0,
        (size_t)B * g->total_repr_mem * sizeof(float), stream));

    /* 2. Copy repr (primary + context) into input buffers (with compression) */
    {
        int n   = g->n_flow;
        int blk = 128, grd = (n + blk - 1) / blk;
        k_copy_blocks_comp<<<grd, blk, 0, stream>>>(
            obj->states[prev].repr_activ.ptr,
            obj->states[cur ].input_activ.ptr,
            obj->g_flow_from.ptr,
            obj->g_flow_size.ptr,
            obj->g_flow_to.ptr,
            n);
    }

    /* Optional repr context copy (feed_context_in_complex_layer) */
    if (obj->cfg.feed_context_in_complex_layer && g->n_repr_flow > 0) {
        int n   = g->n_repr_flow;
        int blk = 128, grd = (n + blk - 1) / blk;
        k_copy_repr_blocks<<<grd, blk, 0, stream>>>(
            obj->states[prev].repr_activ.ptr,
            obj->states[cur ].repr_activ.ptr,
            obj->g_repr_flow_from.ptr,
            obj->g_repr_flow_to.ptr,
            obj->g_repr_flow_size.ptr,
            n);
    }

    /* 3. Wait for frame upload, then distribute patches */
    CUDA_CHECK(cudaStreamWaitEvent(stream, obj->event_upload_done, 0));
    {
        int L0 = obj->cfg.layer_shapes[0];
        int I  = obj->cfg.input_block_size;
        int fp = g->n_frame_units;
        int blk = 128, grd = (fp + blk - 1) / blk;
        k_dist_frame<<<grd, blk, 0, stream>>>(
            obj->gpu_frame.ptr,
            obj->states[cur].input_activ.ptr,
            obj->g_frame_ptr.ptr,
            L0 * I, L0 * I,
            L0, L0,
            I, I,
            0,   /* input_offset = 0 */
            fp);
    }
}

/* ── pvm_forward ─────────────────────────────────────────────────────────── */
/* L0: input -> repr (W0 * input, sigmoid)
 * L1: repr  -> output (W1 * repr, sigmoid)                                  */
void pvm_forward(PVMObject *obj)
{
    cudaStream_t stream = obj->stream_compute;
    PVMGraph *g   = &obj->graph;
    int cur        = cur_slot(obj);
    int B          = obj->batch_size;
    int blk        = 128;

    /* ── Layer 0: W0 * input -> repr ── */
    CUDA_CHECK(cudaMemsetAsync(obj->states[cur].repr_activ.ptr, 0,
        (size_t)B * g->total_repr_mem * sizeof(float), stream));

    {
        int grd = (g->total_threads_L0 + blk - 1) / blk;
        k_dot_fast_set_bias<<<grd, blk, 0, stream>>>(
            obj->weight_main.ptr,
            obj->states[cur].input_activ.ptr,
            obj->states[cur].repr_activ.ptr,
            obj->g_weight_ptr0.ptr,
            obj->g_input_ptr.ptr,
            obj->g_repr_ptr.ptr,
            obj->g_shape0_L0.ptr,
            obj->g_shape1_L0.ptr,
            obj->g_obj_id_L0.ptr,
            obj->g_row_id_L0.ptr,
            g->total_threads_L0);

        int ugrd = (g->total_units + blk - 1) / blk;
        if (!obj->cfg.polynomial)
            k_sigmoid_fast<<<ugrd, blk, 0, stream>>>(
                obj->states[cur].repr_activ.ptr, obj->g_repr_ptr.ptr,
                obj->beta_repr.ptr, obj->g_shape0_L0.ptr, g->total_units);
        else
            k_sigmoid_poly_fast<<<ugrd, blk, 0, stream>>>(
                obj->states[cur].repr_activ.ptr, obj->g_repr_ptr.ptr,
                obj->beta_repr.ptr, obj->g_shape0_L0.ptr, g->total_units);
    }

    /* ── Layer 1: W1 * repr -> output ── */
    CUDA_CHECK(cudaMemsetAsync(obj->states[cur].output_activ.ptr, 0,
        (size_t)B * g->total_input_mem * sizeof(float), stream));

    {
        int grd = (g->total_threads_L1 + blk - 1) / blk;
        k_dot_fast_set_bias<<<grd, blk, 0, stream>>>(
            obj->weight_main.ptr,
            obj->states[cur].repr_activ.ptr,
            obj->states[cur].output_activ.ptr,
            obj->g_weight_ptr1.ptr,
            obj->g_repr_ptr.ptr,
            obj->g_input_ptr.ptr,
            obj->g_shape0_L1.ptr,
            obj->g_shape1_L1.ptr,
            obj->g_obj_id_L1.ptr,
            obj->g_row_id_L1.ptr,
            g->total_threads_L1);

        int ugrd = (g->total_units + blk - 1) / blk;
        if (!obj->cfg.polynomial)
            k_sigmoid_fast<<<ugrd, blk, 0, stream>>>(
                obj->states[cur].output_activ.ptr, obj->g_input_ptr.ptr,
                obj->beta_input.ptr, obj->g_shape0_L1.ptr, g->total_units);
        else
            k_sigmoid_poly_fast<<<ugrd, blk, 0, stream>>>(
                obj->states[cur].output_activ.ptr, obj->g_input_ptr.ptr,
                obj->beta_input.ptr, obj->g_shape0_L1.ptr, g->total_units);
    }
}

/* ── pvm_backward ────────────────────────────────────────────────────────── */
/* Backprop + weight update with momentum.
 * error   = input[cur] - output[lagged]
 * d_out   = sigmoid_der(output[lagged]) * error
 * err_rep = W1^T * d_out
 * d_rep   = sigmoid_der(repr[lagged]) * err_rep
 * dW1     = momentum*dW1 + lr * outer(d_out, repr[lagged])
 * dW0     = momentum*dW0 + lr * outer(d_rep, input[lagged])
 * W      += dW                                                               */
void pvm_backward(PVMObject *obj)
{
    cudaStream_t stream = obj->stream_compute;
    PVMGraph *g   = &obj->graph;
    int cur        = cur_slot(obj);
    int lagged     = lagged_slot(obj);
    int B          = obj->batch_size;
    int U          = g->total_units;
    int blk        = 128;

    /* ── error = input[cur] - output[lagged] ── */
    {
        /* output_error[lagged] = -output_activ[lagged], then += input[cur] */
        CUDA_CHECK(cudaMemcpyAsync(
            obj->states[lagged].output_error.ptr,
            obj->states[lagged].output_activ.ptr,
            (size_t)B * g->total_input_mem * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));

        float neg_one = -1.0f, one = 1.0f;
        int   nn      = (int)((size_t)B * g->total_input_mem);
        CUBLAS_CHECK(cublasSscal(s_cublas, nn, &neg_one,
            obj->states[lagged].output_error.ptr, 1));
        CUBLAS_CHECK(cublasSaxpy(s_cublas, nn, &one,
            obj->states[cur].input_activ.ptr, 1,
            obj->states[lagged].output_error.ptr, 1));
    }

    /* ── delta_output = sigmoid_der(output[lagged]) * error ── */
    {
        int ugrd = (U + blk - 1) / blk;
        if (!obj->cfg.polynomial)
            k_sigmoid_der_mul<<<ugrd, blk, 0, stream>>>(
                obj->states[lagged].output_activ.ptr,
                obj->states[lagged].output_error.ptr,
                obj->states[lagged].output_delta.ptr,
                obj->g_input_ptr.ptr, obj->g_input_ptr.ptr, obj->g_input_ptr.ptr,
                obj->g_shape0_L1.ptr, U);
        else
            k_sigmoid_poly_der_mul<<<ugrd, blk, 0, stream>>>(
                obj->states[lagged].output_activ.ptr,
                obj->states[lagged].output_error.ptr,
                obj->states[lagged].output_delta.ptr,
                obj->g_input_ptr.ptr, obj->g_input_ptr.ptr, obj->g_input_ptr.ptr,
                obj->g_shape0_L1.ptr, U);
    }

    /* ── W1^T * delta_output -> repr_error ── */
    {
        CUDA_CHECK(cudaMemsetAsync(obj->states[lagged].repr_error.ptr, 0,
            (size_t)B * g->total_repr_mem * sizeof(float), stream));

        int grd = (g->total_threads_L1 + blk - 1) / blk;
        k_dot_transpose_fast<<<grd, blk, 0, stream>>>(
            obj->weight_main.ptr,
            obj->weight_cache.ptr,
            obj->states[lagged].output_delta.ptr,
            obj->g_weight_ptr1.ptr,
            obj->g_input_ptr.ptr,
            obj->g_shape0_L1.ptr,
            obj->g_shape1_L1.ptr,
            obj->g_obj_id_L1.ptr,
            obj->g_row_id_L1.ptr,
            g->total_threads_L1);

        k_sum_dot_transpose<<<grd, blk, 0, stream>>>(
            obj->weight_cache.ptr,
            obj->states[lagged].repr_error.ptr,
            obj->g_weight_ptr1.ptr,
            obj->g_repr_ptr.ptr,
            obj->g_shape0_L1.ptr,
            obj->g_shape1_L1.ptr,
            obj->g_obj_id_L1.ptr,
            obj->g_row_id_L1.ptr,
            g->total_threads_L1);
    }

    /* ── delta_repr = sigmoid_der(repr[lagged]) * repr_error ── */
    {
        int ugrd = (U + blk - 1) / blk;
        if (!obj->cfg.polynomial)
            k_sigmoid_der_mul<<<ugrd, blk, 0, stream>>>(
                obj->states[lagged].repr_activ.ptr,
                obj->states[lagged].repr_error.ptr,
                obj->states[lagged].repr_delta.ptr,
                obj->g_repr_ptr.ptr, obj->g_repr_ptr.ptr, obj->g_repr_ptr.ptr,
                obj->g_shape1_L1.ptr, U);
        else
            k_sigmoid_poly_der_mul<<<ugrd, blk, 0, stream>>>(
                obj->states[lagged].repr_activ.ptr,
                obj->states[lagged].repr_error.ptr,
                obj->states[lagged].repr_delta.ptr,
                obj->g_repr_ptr.ptr, obj->g_repr_ptr.ptr, obj->g_repr_ptr.ptr,
                obj->g_shape1_L1.ptr, U);
    }

    /* ── Weight updates ── */
    int next_buf = (obj->buf_idx + 1) % 2;

    /* dW1 = momentum * dW1_prev + lr * outer(delta_output, repr[lagged]) */
    {
        int grd = (g->total_threads_L1 + blk - 1) / blk;
        k_outer_update<<<grd, blk, 0, stream>>>(
            obj->states[lagged].output_delta.ptr,
            obj->states[lagged].repr_activ.ptr,
            obj->dweight[next_buf].ptr,
            obj->dweight[obj->buf_idx].ptr,
            obj->g_input_ptr.ptr,
            obj->g_repr_ptr.ptr,
            obj->g_weight_ptr1.ptr,
            obj->g_shape0_L1.ptr,
            obj->g_shape1_L1.ptr,
            obj->lr_arr.ptr,
            obj->momentum_arr.ptr,
            obj->g_obj_id_L1.ptr,
            obj->g_row_id_L1.ptr,
            g->total_threads_L1);
    }

    /* dW0 = momentum * dW0_prev + lr * outer(delta_repr, input[lagged]) */
    {
        int grd = (g->total_threads_L0 + blk - 1) / blk;
        k_outer_update<<<grd, blk, 0, stream>>>(
            obj->states[lagged].repr_delta.ptr,
            obj->states[lagged].input_activ.ptr,
            obj->dweight[next_buf].ptr,
            obj->dweight[obj->buf_idx].ptr,
            obj->g_repr_ptr.ptr,
            obj->g_input_ptr.ptr,
            obj->g_weight_ptr0.ptr,
            obj->g_shape0_L0.ptr,
            obj->g_shape1_L0.ptr,
            obj->lr_arr.ptr,
            obj->momentum_arr.ptr,
            obj->g_obj_id_L0.ptr,
            obj->g_row_id_L0.ptr,
            g->total_threads_L0);
    }

    /* W += dW */
    {
        int n   = g->total_weights;
        int grd = (n + 256 - 1) / 256;
        k_weight_add<<<grd, 256, 0, stream>>>(
            obj->weight_main.ptr, obj->dweight[obj->buf_idx].ptr, n);
    }

    obj->buf_idx = (obj->buf_idx + 1) % 2;
    obj->step++;
}

/* ── CUDA Graph capture ──────────────────────────────────────────────────── */
/* Captures forward+backward into a CUDA graph for zero-CPU-overhead replay.
 * Must be called after a few warm-up steps so lazy state is initialized.    */
void pvm_build_cuda_graph(PVMObject *obj)
{
    if (obj->graph_captured) return;

    /* Warm-up: run a few eager steps to initialise any lazy CUDA state */
    for (int i = 0; i < 3; ++i) {
        pvm_forward(obj);
        pvm_backward(obj);
        CUDA_CHECK(cudaStreamSynchronize(obj->stream_compute));
        obj->step--;  /* don't count warm-up steps */
    }

    /* Begin CUDA graph capture on compute stream */
    CUDA_CHECK(cudaStreamBeginCapture(obj->stream_compute,
                                       cudaStreamCaptureModeGlobal));
    pvm_forward(obj);
    pvm_backward(obj);
    CUDA_CHECK(cudaStreamEndCapture(obj->stream_compute, &obj->cuda_graph));
    CUDA_CHECK(cudaGraphInstantiate(&obj->cuda_graph_exec,
                                     obj->cuda_graph, NULL, NULL, 0));
    obj->graph_captured = 1;
    printf("CUDA Graph captured successfully.\n");
}

/* Replay the captured CUDA graph */
void pvm_run_cuda_graph(PVMObject *obj)
{
    if (!obj->graph_captured) pvm_build_cuda_graph(obj);
    CUDA_CHECK(cudaGraphLaunch(obj->cuda_graph_exec, obj->stream_compute));
    /* step is incremented inside pvm_backward; increment manually when using graph */
    obj->step++;
}

/* ── pvm_update_learning_rate ───────────────────────────────────────────── */
/* Mirrors Python learning-rate schedule:
 *   - Enable one layer per delay_each_layer_learning steps
 *   - Switch to intermediate LR at delay_intermediate_learning_rate
 *   - Switch to final       LR at delay_final_learning_rate              */
void pvm_update_learning_rate(PVMObject *obj)
{
    const PVMConfig *cfg = &obj->cfg;
    PVMGraph        *g   = &obj->graph;
    int U = g->total_units;
    int changed = 0;

    float *lr_h  = (float *)malloc((size_t)U * sizeof(float));
    float *mom_h = (float *)malloc((size_t)U * sizeof(float));
    CUDA_CHECK(cudaMemcpy(lr_h,  obj->lr_arr.ptr,       (size_t)U*sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mom_h, obj->momentum_arr.ptr, (size_t)U*sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* Enable layers progressively */
    if (cfg->delay_each_layer_learning > 0 &&
        obj->step % cfg->delay_each_layer_learning == 0)
    {
        long long layer_to_enable = obj->step / cfg->delay_each_layer_learning;
        if (layer_to_enable < (long long)cfg->num_layers) {
            int begin = g->layer_ptrs[layer_to_enable];
            int end   = begin + cfg->layer_shapes[layer_to_enable]
                              * cfg->layer_shapes[layer_to_enable];
            printf("Enabling layer %lld (units %d..%d)  lr=%.5f\n",
                   layer_to_enable, begin, end, cfg->initial_learning_rate);
            for (int i = begin; i < end; ++i) {
                lr_h [i] = cfg->initial_learning_rate;
                mom_h[i] = cfg->momentum;
            }
            changed = 1;
        }
    }

    if (obj->step == cfg->delay_intermediate_learning_rate) {
        printf("Setting intermediate LR %.6f\n", cfg->intermediate_learning_rate);
        for (int i = 0; i < U; ++i) lr_h[i] = cfg->intermediate_learning_rate;
        changed = 1;
    }

    if (obj->step == cfg->delay_final_learning_rate) {
        printf("Setting final LR %.6f\n", cfg->final_learning_rate);
        for (int i = 0; i < U; ++i) lr_h[i] = cfg->final_learning_rate;
        changed = 1;
    }

    if (changed) {
        CUDA_CHECK(cudaMemcpy(obj->lr_arr.ptr,       lr_h,
                              (size_t)U*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(obj->momentum_arr.ptr, mom_h,
                              (size_t)U*sizeof(float), cudaMemcpyHostToDevice));
        /* Invalidate CUDA graph because lr_arr device pointer values changed */
        if (obj->graph_captured) {
            cudaGraphExecDestroy(obj->cuda_graph_exec);
            cudaGraphDestroy(obj->cuda_graph);
            obj->cuda_graph_exec = NULL;
            obj->cuda_graph      = NULL;
            obj->graph_captured  = 0;
        }
    }

    free(lr_h);
    free(mom_h);
}

/* ── pvm_save / pvm_load ─────────────────────────────────────────────────── */
/* Binary format: magic(4) + step(8) + n_weights(4) + weights(4*n)           */
int pvm_save(const PVMObject *obj, const char *path)
{
    CUDA_CHECK(cudaStreamSynchronize(obj->stream_compute));
    int nw = obj->graph.total_weights;
    float *wh = (float *)malloc((size_t)nw * sizeof(float));
    CUDA_CHECK(cudaMemcpy(wh, obj->weight_main.ptr,
                          (size_t)nw * sizeof(float), cudaMemcpyDeviceToHost));

    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "pvm_save: cannot open '%s'\n", path); free(wh); return -1; }

    uint32_t magic = 0x50564D43u; /* "PVMC" */
    fwrite(&magic,    4, 1, fp);
    fwrite(&obj->step, sizeof(obj->step), 1, fp);
    fwrite(&nw,        sizeof(int), 1, fp);
    fwrite(wh,         sizeof(float), (size_t)nw, fp);
    fclose(fp);
    free(wh);
    printf("Saved model to '%s' (step %lld)\n", path, (long long)obj->step);
    return 0;
}

int pvm_load(PVMObject *obj, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "pvm_load: cannot open '%s'\n", path); return -1; }

    uint32_t magic;
    (void)fread(&magic, 4, 1, fp);
    if (magic != 0x50564D43u) {
        fprintf(stderr, "pvm_load: bad magic in '%s'\n", path);
        fclose(fp); return -1;
    }
      (void)fread(&obj->step, sizeof(obj->step), 1, fp);
      int nw; (void)fread(&nw, sizeof(int), 1, fp);
    if (nw != obj->graph.total_weights) {
        fprintf(stderr, "pvm_load: weight count mismatch (%d vs %d)\n",
                nw, obj->graph.total_weights);
        fclose(fp); return -1;
    }
    float *wh = (float *)malloc((size_t)nw * sizeof(float));
      (void)fread(wh, sizeof(float), (size_t)nw, fp);
    fclose(fp);
    CUDA_CHECK(cudaMemcpy(obj->weight_main.ptr, wh,
                          (size_t)nw * sizeof(float), cudaMemcpyHostToDevice));
    free(wh);
    printf("Loaded model from '%s' (step %lld)\n", path, (long long)obj->step);
    return 0;
}

/* ── Readout helpers ─────────────────────────────────────────────────────── */
void pvm_pop_prediction(const PVMObject *obj, float *out_host, int b)
{
    cudaStream_t s = obj->stream_compute;
    int cur  = cur_slot(obj);
    int L0   = obj->cfg.layer_shapes[0];
    int I    = obj->cfg.input_block_size;
    int H    = L0 * I;
    size_t frame_sz = (size_t)H * H * obj->cfg.input_channels * sizeof(float);

    float *gpu_out;
    CUDA_CHECK(cudaMalloc(&gpu_out, frame_sz));
    CUDA_CHECK(cudaMemsetAsync(gpu_out, 0, frame_sz, s));

    int fp  = obj->graph.n_frame_units;
    int blk = 128, grd = (fp + blk - 1) / blk;
    k_collect_frame<<<grd, blk, 0, s>>>(
        gpu_out,
        obj->states[cur].output_activ.ptr + (size_t)b * obj->graph.total_input_mem,
        obj->g_frame_ptr.ptr,
        H, H, L0, L0, I, I, 0, fp);

    CUDA_CHECK(cudaMemcpyAsync(out_host, gpu_out, frame_sz, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(gpu_out));
}

void pvm_pop_layer(const PVMObject *obj, uint8_t *out_host, int layer, int b)
{
    cudaStream_t s = obj->stream_compute;
    int cur = cur_slot(obj);
    int L   = obj->cfg.layer_shapes[layer];
    int H   = L * obj->cfg.hidden_block_size;
    size_t sz = (size_t)H * H * sizeof(unsigned int);

    unsigned int *gpu_out;
    CUDA_CHECK(cudaMalloc(&gpu_out, sz));
    CUDA_CHECK(cudaMemsetAsync(gpu_out, 0, sz, s));

    int patches = L * L;
    int blk = 128, grd = (patches + blk - 1) / blk;
    k_collect_activ<<<grd, blk, 0, s>>>(
        gpu_out,
        obj->states[cur].repr_activ.ptr + (size_t)b * obj->graph.total_repr_mem,
        obj->g_repr_ptr.ptr,
        H, H, L, L,
        obj->cfg.hidden_block_size, obj->cfg.hidden_block_size,
        obj->graph.layer_ptrs[layer],
        patches);

    unsigned int *tmp = (unsigned int *)malloc(H * H * sizeof(unsigned int));
    CUDA_CHECK(cudaMemcpyAsync(tmp, gpu_out, sz, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(gpu_out));
    for (int i = 0; i < H * H; ++i) out_host[i] = (uint8_t)(tmp[i] & 0xFFu);
    free(tmp);
}
