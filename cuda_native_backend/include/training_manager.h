#pragma once
// Training Manager - orchestrates the training loop
// Equivalent of Python's manager.py

#include "pvm_object.cuh"
#include "data_provider.h"
#include <memory>
#include <string>
#include <chrono>

struct TrainingManagerOptions {
    bool   display       = false;
    bool   snapshot      = false;
    bool   use_cuda_graph = true;
    int    save_every    = 100000;   // steps between checkpoints
    std::string save_prefix = "./Sim";
    std::string model_name  = "pvm";
};

class TrainingManager {
public:
    using Options = TrainingManagerOptions;

    TrainingManager(std::shared_ptr<PVMObject>    pvm,
                    std::shared_ptr<DataProvider>  data,
                    const Options& opts = Options{});

    void run(long long steps);

private:
    void step_once();
    void print_fps();
    void save_state();
    void upload_frames_to_gpu();

    std::shared_ptr<PVMObject>   pvm_;
    std::shared_ptr<DataProvider> data_;
    Options opts_;

    long long counter_   = 0;
    long long total_steps_ = 0;
    bool      graph_built_ = false;
    float fps_           = 0.f;
    float inst_fps_      = 0.f;

    using Clock = std::chrono::steady_clock;
    Clock::time_point t_start_;
    Clock::time_point t_last_;
    long long frames_since_last_ = 0;

    // Host-pinned staging buffer for frame batches
    float* pinned_frame_ = nullptr;
    size_t pinned_bytes_  = 0;

    cudaStream_t stream_upload_  = nullptr;
    cudaStream_t stream_compute_ = nullptr;
    cudaEvent_t  ev_upload_      = nullptr;
};
