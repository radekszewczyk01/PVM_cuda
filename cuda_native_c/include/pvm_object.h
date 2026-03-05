/* pvm_object.h – PVM training object (C-callable interface)
 * All CUDA implementation lives in pvm_object.cu.
 * This header exposes only C types and function signatures so that training_manager.c
 * and main.c can be compiled as plain C (cc) rather than nvcc.
 */
#ifndef PVM_OBJECT_H
#define PVM_OBJECT_H

#include "pvm_config.h"
#include "pvm_graph.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque GPU buffer handle ───────────────────────────────────────────────── */
typedef struct {
    float *ptr;
    size_t bytes;
} GpuBuf;

typedef struct {
    int   *ptr;
    size_t bytes;
} GpuIntBuf;

/* ── Per-timestep activation state ─────────────────────────────────────────── */
typedef struct {
    GpuBuf input_activ,  input_error,  input_delta;
    GpuBuf output_activ, output_error, output_delta;
    GpuBuf repr_activ,   repr_error,   repr_delta;
} StepState;

/* Maximum sequence buffer length (= graph.sequence_length = 3) */
#define PVM_MAX_SEQ 8

/* ── Main PVMObject struct ──────────────────────────────────────────────────── */
typedef struct {
    PVMConfig cfg;
    PVMGraph  graph;
    int       batch_size;
    long long step;
    int       buf_idx;       /* momentum double-buffer index */

    /* ── Weights ── */
    GpuBuf    weight_main;
    GpuBuf    dweight[2];    /* momentum double buffer */
    GpuBuf    weight_cache;  /* scratch for backward transpose */

    /* ── Per-timestep states (sequence_length slots) ── */
    StepState states[PVM_MAX_SEQ];

    /* ── Gain (beta) arrays ── */
    GpuBuf    beta_input;
    GpuBuf    beta_repr;

    /* ── Learning-rate / momentum arrays (per unit) ── */
    GpuBuf    lr_arr;
    GpuBuf    momentum_arr;

    /* ── Frame buffer (GPU) ── */
    GpuBuf    gpu_frame;

    /* ── Per-unit metadata on GPU ── */
    GpuIntBuf g_weight_ptr0, g_weight_ptr1;
    GpuIntBuf g_shape0_L0,   g_shape1_L0;
    GpuIntBuf g_shape0_L1,   g_shape1_L1;
    GpuIntBuf g_input_ptr,   g_repr_ptr;
    GpuIntBuf g_obj_id_L0,   g_row_id_L0;
    GpuIntBuf g_obj_id_L1,   g_row_id_L1;

    /* ── Flow pointer arrays on GPU ── */
    GpuIntBuf g_flow_from,      g_flow_to,      g_flow_size;
    GpuIntBuf g_repr_flow_from, g_repr_flow_to, g_repr_flow_size;
    GpuIntBuf g_shift_from,     g_shift_to,     g_shift_size;
    GpuIntBuf g_frame_ptr,      g_frame_ptr_size;

    /* ── CUDA streams & events ── */
    cudaStream_t stream_compute;
    cudaStream_t stream_upload;
    cudaEvent_t  event_upload_done;

    /* ── CUDA graph for forward+backward capture ── */
    cudaGraph_t     cuda_graph;
    cudaGraphExec_t cuda_graph_exec;
    int             graph_captured;    /* bool: has graph been captured? */

    /* ── Pinned host frame staging buffer ── */
    float  *pinned_frame;
    size_t  pinned_frame_bytes;
} PVMObject;

/* ── Lifecycle ──────────────────────────────────────────────────────────────── */

/* Allocate and initialise a PVMObject.  Returns pointer (must call pvm_object_destroy). */
PVMObject *pvm_object_create(const PVMConfig *cfg);

/* Free all GPU memory and destroy object. */
void pvm_object_destroy(PVMObject *obj);

/* ── Main interface ─────────────────────────────────────────────────────────── */

/* Push a batch of float32 frames [batch * H * W * C] into the model.
 * frame_host must be in pinned or host-accessible memory.              */
void pvm_push_input(PVMObject *obj, const float *frame_host);

/* Execute one forward pass (L0: input->repr, L1: repr->output). */
void pvm_forward(PVMObject *obj);

/* Execute one backward pass + weight update with momentum. */
void pvm_backward(PVMObject *obj);

/* Capture forward+backward into a CUDA graph (call once after warm-up). */
void pvm_build_cuda_graph(PVMObject *obj);

/* Replay the captured CUDA graph (much lower CPU overhead than eager execution). */
void pvm_run_cuda_graph(PVMObject *obj);

/* ── Learning-rate schedule ─────────────────────────────────────────────────── */
void pvm_update_learning_rate(PVMObject *obj);

/* ── Checkpoint I/O ─────────────────────────────────────────────────────────── */
int pvm_save(const PVMObject *obj, const char *path);
int pvm_load(PVMObject *obj, const char *path);

/* ── Readout helpers ────────────────────────────────────────────────────────── */
/* Download the prediction frame for batch entry b into out_host (float32 RGB). */
void pvm_pop_prediction(const PVMObject *obj, float *out_host, int b);

/* Download layer repr as uint8 grayscale image for batch entry b. */
void pvm_pop_layer(const PVMObject *obj, uint8_t *out_host, int layer, int b);

#ifdef __cplusplus
}
#endif

#endif /* PVM_OBJECT_H */
