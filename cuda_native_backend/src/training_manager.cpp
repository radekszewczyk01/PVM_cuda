// Training Manager
#include "training_manager.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
        exit(1); \
    } \
} while(0)

TrainingManager::TrainingManager(
    std::shared_ptr<PVMObject>   pvm,
    std::shared_ptr<DataProvider> data,
    const Options& opts)
    : pvm_(pvm), data_(data), opts_(opts)
{
    CUDA_CHECK(cudaStreamCreate(&stream_upload_));
    CUDA_CHECK(cudaStreamCreate(&stream_compute_));
    CUDA_CHECK(cudaEventCreate(&ev_upload_));

    // Allocate pinned host frame buffer (batch * H * W * C)
    int    B  = pvm_->batch_size();
    int    sz = pvm_->input_size();
    size_t ch = pvm_->config().input_channels;
    pinned_bytes_ = (size_t)B * sz * sz * ch * sizeof(float);
    CUDA_CHECK(cudaMallocHost(&pinned_frame_, pinned_bytes_));
}

void TrainingManager::run(long long steps)
{
    total_steps_ = steps;
    t_start_ = t_last_ = Clock::now();

    // Pre-fill: reset data and prime the step counter
    data_->advance();

    // Capture CUDA graph after a few warm-up steps
    graph_built_ = false;

    for (counter_ = 0; counter_ < steps; ++counter_) {
        step_once();

        if (!graph_built_ && opts_.use_cuda_graph && pvm_->step() > 10) {
            pvm_->build_cuda_graph();
            graph_built_ = true;
        }

        print_fps();

        if (pvm_->step() > 1 && pvm_->step() % opts_.save_every == 0)
            save_state();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_compute_));
}

void TrainingManager::step_once()
{
    // 1. Upload frames (batch) to pinned buffer
    upload_frames_to_gpu();

    // 2. Update learning rate (host-side schedule)
    pvm_->update_learning_rate();

    // 3. Forward + backward (using CUDA Graph if built)
    pvm_->push_input_gpu(pinned_frame_, stream_compute_);
    if (graph_built_ && opts_.use_cuda_graph) {
        pvm_->run_cuda_graph();
    } else {
        pvm_->forward_gpu(stream_compute_);
        pvm_->backward_gpu(stream_compute_);
    }

    frames_since_last_++;
}

void TrainingManager::upload_frames_to_gpu()
{
    // For each batch element, get the next frame and copy into pinned buffer
    int    sz = pvm_->input_size();
    size_t ch = pvm_->config().input_channels;
    size_t frame_floats = (size_t)sz * sz * ch;

    for (int b = 0; b < pvm_->batch_size(); ++b) {
        data_->advance();
        Frame fr = data_->get_next();
        if (fr.image.empty()) continue;

        // Ensure contiguous float32 RGB
        cv::Mat f32;
        if (fr.image.type() != CV_32FC3) fr.image.convertTo(f32, CV_32FC3, 1.0f/255.f);
        else f32 = fr.image;

        // Copy into pinned buffer at batch offset
        float* dst = pinned_frame_ + b * frame_floats;
        memcpy(dst, f32.data, frame_floats * sizeof(float));
    }
}

void TrainingManager::print_fps()
{
    auto now = Clock::now();
    double elapsed = std::chrono::duration<double>(now - t_start_).count();
    double since_last = std::chrono::duration<double>(now - t_last_).count();

    fps_ = (float)(pvm_->step() / elapsed);
    if (since_last > 1.0) {
        inst_fps_ = (float)(frames_since_last_ / since_last);
        double pct      = total_steps_ > 0 ? 100.0 * counter_ / total_steps_ : 0.0;
        long long left  = total_steps_ - counter_;
        double eta_s    = (inst_fps_ > 0.f) ? left / (double)inst_fps_ : 0.0;
        long long eta_h = (long long)eta_s / 3600;
        long long eta_m = ((long long)eta_s % 3600) / 60;
        long long eta_sec = (long long)eta_s % 60;
        printf("\r%lld/%lld (%.1f%%) | fps: %.1f avg / %.1f inst | ETA: %lluh%02llum%02llus   ",
               counter_, total_steps_, pct, fps_, inst_fps_,
               eta_h, eta_m, eta_sec);
        fflush(stdout);
        t_last_ = now;
        frames_since_last_ = 0;
    }
}

void TrainingManager::save_state()
{
    char path[512];
    snprintf(path, sizeof(path), "%s_%s_%09lld.bin",
             opts_.save_prefix.c_str(),
             opts_.model_name.c_str(),
             pvm_->step());
    pvm_->save(path);
}
