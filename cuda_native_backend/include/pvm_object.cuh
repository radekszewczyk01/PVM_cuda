#pragma once
// PVM Object - main training object
// C++ equivalent of Python's PVM_object in sequence_learner.py
// Key improvements:
//   - cuBLAS for large batched matrix ops
//   - CUDA Streams: overlap frame dispatch and compute
//   - CUDA Graphs: capture forward+backward loop, replay without CPU overhead
//   - Multi-sequence batch training (B independent sequences)

#include "pvm_config.h"
#include "pvm_graph.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <memory>

// RAII wrapper for a flat GPU buffer
struct GpuBuffer {
    float* ptr = nullptr;
    size_t bytes = 0;

    GpuBuffer() = default;
    GpuBuffer(size_t n_floats);
    ~GpuBuffer();
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& o) noexcept;
    GpuBuffer& operator=(GpuBuffer&& o) noexcept;

    void fill(float val, cudaStream_t s = 0);
    void upload(const float* host, cudaStream_t s = 0);
    void download(float* host, cudaStream_t s = 0) const;
};

struct GpuIntBuffer {
    int* ptr = nullptr;
    size_t bytes = 0;

    GpuIntBuffer() = default;
    explicit GpuIntBuffer(const std::vector<int>& v);
    ~GpuIntBuffer();
    GpuIntBuffer(const GpuIntBuffer&) = delete;
    GpuIntBuffer& operator=(const GpuIntBuffer&) = delete;
    GpuIntBuffer(GpuIntBuffer&& o) noexcept;
    GpuIntBuffer& operator=(GpuIntBuffer&& o) noexcept;
};

// Per-step activation state (input / repr / output + error + delta)
// Holds batch_size independent states for each time-slot in the sequence buffer
struct StepState {
    GpuBuffer input_activ;   // [batch * total_input_mem]
    GpuBuffer input_error;
    GpuBuffer input_delta;

    GpuBuffer output_activ;  // [batch * total_input_mem]
    GpuBuffer output_error;
    GpuBuffer output_delta;

    GpuBuffer repr_activ;    // [batch * total_repr_mem]
    GpuBuffer repr_error;
    GpuBuffer repr_delta;
};

class PVMObject {
public:
    explicit PVMObject(const PVMConfig& cfg, int batch_size = 1);
    ~PVMObject();

    // ── Main training interface ──────────────────────────────────────────────
    // Push a batch of frames (float32 RGB, [batch, H, W, 3])
    void push_input_gpu(const float* frame_batch_host, cudaStream_t stream = 0);

    void forward_gpu(cudaStream_t stream = 0);
    void backward_gpu(cudaStream_t stream = 0);

    // Capture a full forward+backward into a CUDA Graph (call once, then replay)
    void build_cuda_graph();
    void run_cuda_graph();  // replay captured graph

    // ── Learning rate schedule ───────────────────────────────────────────────
    void update_learning_rate();

    // ── Save / load ──────────────────────────────────────────────────────────
    void save(const std::string& path) const;
    void load(const std::string& path);

    // ── Accessors ────────────────────────────────────────────────────────────
    int   input_size()   const { return cfg_.input_size(); }
    int   batch_size()   const { return batch_size_; }
    long long step()     const { return step_; }
    float fps()          const { return fps_; }

    // Download prediction frame for batch entry b (float32 RGB)
    void pop_prediction(float* out_host, int b = 0, cudaStream_t s = 0) const;

    // Download layer activation as uint8 grayscale image (for visualization)
    void pop_layer(uint8_t* out_host, int layer, int b = 0, cudaStream_t s = 0) const;

    const PVMConfig& config() const { return cfg_; }
    const PVMGraph&  graph()  const { return graph_; }

private:
    void allocate_gpu_memory();
    void upload_graph_to_gpu();
    void zero_activations(cudaStream_t s);

    // Current and previous time-buffer indices
    int cur_slot()  const { return (int)(step_ % graph_.sequence_length); }
    int prev_slot() const { return (int)((step_ - 1 + graph_.sequence_length) % graph_.sequence_length); }
    int lagged_slot() const {
        return (int)((step_ - graph_.sequence_interval + graph_.sequence_length) % graph_.sequence_length);
    }

    PVMConfig cfg_;
    PVMGraph  graph_;
    int       batch_size_;
    long long step_    = 0;
    int       buf_idx_ = 0;  // momentum buffer flip index
    float     fps_     = 0.f;
    float     inst_fps_ = 0.f;

    cublasHandle_t cublas_;

    // ── Weights (shared across batch) ───────────────────────────────────────
    GpuBuffer weight_main_;    // [total_weights]
    GpuBuffer dweight_[2];     // [total_weights] x2 (double buffer for momentum)
    GpuBuffer weight_cache_;   // [total_weights] temporary for backward

    // ── Per-timestep activation states [sequence_length] ────────────────────
    std::vector<std::unique_ptr<StepState>> states_;

    // ── Beta (gain) arrays ───────────────────────────────────────────────────
    GpuBuffer beta_input_;   // [total_input_mem]
    GpuBuffer beta_repr_;    // [total_repr_mem]

    // ── Per-unit metadata arrays (on GPU) ───────────────────────────────────
    GpuIntBuffer g_weight_ptr0_, g_weight_ptr1_;
    GpuIntBuffer g_shape0_L0_,   g_shape1_L0_;
    GpuIntBuffer g_shape0_L1_,   g_shape1_L1_;
    GpuIntBuffer g_input_ptr_,   g_repr_ptr_;
    GpuIntBuffer g_obj_id_L0_,   g_row_id_L0_;
    GpuIntBuffer g_obj_id_L1_,   g_row_id_L1_;

    // ── Flow pointers (on GPU) ──────────────────────────────────────────────
    GpuIntBuffer g_flow_from_,  g_flow_to_,  g_flow_size_;
    GpuIntBuffer g_repr_flow_from_, g_repr_flow_to_, g_repr_flow_size_;
    GpuIntBuffer g_shift_from_, g_shift_to_, g_shift_size_;
    GpuIntBuffer g_frame_ptr_,  g_frame_ptr_size_;

    // ── Learning rate / momentum per unit ───────────────────────────────────
    GpuBuffer lr_arr_;       // [total_units]
    GpuBuffer momentum_arr_; // [total_units]

    // ── CUDA Graph for main loop ─────────────────────────────────────────────
    bool  graph_captured_ = false;
    cudaGraph_t     cuda_graph_     = nullptr;
    cudaGraphExec_t cuda_graph_exec_= nullptr;

    // Pinned host buffer for frame upload
    float* pinned_frame_ = nullptr;
    size_t pinned_frame_bytes_ = 0;

    // Temporary GPU frame buffer
    GpuBuffer gpu_frame_;

    // CUDA Streams: one for data upload, one for compute
    cudaStream_t stream_compute_  = nullptr;
    cudaStream_t stream_upload_   = nullptr;
    cudaEvent_t  event_upload_done_ = nullptr;
};
