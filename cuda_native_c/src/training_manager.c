/* training_manager.c – Training loop (plain C99)
 * Links against libcuda via pvm_object.cu symbols (no direct CUDA calls here).
 */
#include "training_manager.h"
#include "pvm_object.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <cuda_runtime.h>

/* ── SIGINT / SIGTERM handler ────────────────────────────────────────────── */
static volatile sig_atomic_t g_stop = 0;
static void handle_stop(int sig) { (void)sig; g_stop = 1; }

/* ── CUDA check (for StreamSynchronize) ─────────────────────────────────── */
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error '%s' at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/* ── Monotonic nanosecond timer ─────────────────────────────────────────── */
static long long now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ── Default options ─────────────────────────────────────────────────────── */
void tm_default_options(TrainingManagerOptions *opts)
{
    memset(opts, 0, sizeof(*opts));
    opts->use_cuda_graph  = 1;
    opts->save_every      = 100000;
    opts->warmup_steps    = 20;
    opts->print_every_ms  = 1000;
    strncpy(opts->save_prefix, "pvm_save",  sizeof(opts->save_prefix)-1);
    strncpy(opts->model_name,  "model",     sizeof(opts->model_name)-1);
}

/* ── tm_init ─────────────────────────────────────────────────────────────── */
void tm_init(TrainingManager *tm,
             PVMObject       *pvm,
             DataProvider    *data,
             const TrainingManagerOptions *opts)
{
    memset(tm, 0, sizeof(*tm));
    tm->pvm   = pvm;
    tm->data  = data;
    if (opts) tm->opts = *opts;
    else      tm_default_options(&tm->opts);
}

/* ── Upload one batch of frames from data provider to pinned buffer ─────── */
static void upload_batch(TrainingManager *tm)
{
    PVMObject    *pvm  = tm->pvm;
    DataProvider *data = tm->data;
    int B   = pvm->batch_size;
    int W   = pvm->cfg.layer_shapes[0] * pvm->cfg.input_block_size;
    int H   = W;
    int C   = pvm->cfg.input_channels;
    size_t frame_floats = (size_t)W * H * C;

    for (int b = 0; b < B; ++b) {
        data->advance(data);
        const float *fr = data->get_frame(data);
        if (!fr) continue;
        memcpy(pvm->pinned_frame + (size_t)b * frame_floats,
               fr, frame_floats * sizeof(float));
    }
}

/* ── Print FPS / progress line ───────────────────────────────────────────── */
static void print_progress(TrainingManager *tm)
{
    long long now     = now_ns();
    long long elapsed = now - tm->t_start_ns;
    long long since   = now - tm->t_last_ns;
    long long since_ms = since / 1000000LL;

    if (since_ms < tm->opts.print_every_ms) return;

    double elapsed_s = (double)elapsed * 1e-9;
    double since_s   = (double)since   * 1e-9;
    tm->fps_avg  = (elapsed_s > 0) ? (double)tm->pvm->step / elapsed_s : 0.0;
    tm->fps_inst = (since_s   > 0) ? (double)tm->frames_since_last / since_s : 0.0;

    long long left = tm->total_steps - tm->counter;
    double eta_s   = (tm->fps_inst > 0.0) ? (double)left / tm->fps_inst : 0.0;
    long long eta_h = (long long)eta_s / 3600;
    long long eta_m = ((long long)eta_s % 3600) / 60;
    long long eta_sec = (long long)eta_s % 60;
    double pct = (tm->total_steps > 0)
               ? 100.0 * (double)tm->counter / (double)tm->total_steps : 0.0;

    printf("\r%lld/%lld (%.1f%%) | fps: %.1f avg / %.1f inst | ETA: %lldh%02lldm%02llds   ",
           (long long)tm->counter, (long long)tm->total_steps, pct,
           tm->fps_avg, tm->fps_inst,
           eta_h, eta_m, eta_sec);
    fflush(stdout);

    tm->t_last_ns      = now;
    tm->frames_since_last = 0;
}

/* ── Single training step ─────────────────────────────────────────────────── */
static void tm_step(TrainingManager *tm)
{
    /* 1. Load frames into pinned buffer */
    upload_batch(tm);

    /* 2. Update learning-rate schedule */
    pvm_update_learning_rate(tm->pvm);

    /* 3. Push frames into GPU memory (handles temporal shift + frame dist) */
    pvm_push_input(tm->pvm, tm->pvm->pinned_frame);

    /* 4. Forward + backward (via CUDA Graph or eager kernels) */
    if (tm->graph_built && tm->opts.use_cuda_graph) {
        /* Graph replay: step counter is incremented inside run_cuda_graph */
        pvm_run_cuda_graph(tm->pvm);
    } else {
        pvm_forward(tm->pvm);
        pvm_backward(tm->pvm);
    }

    tm->frames_since_last++;
}

/* ── tm_run ──────────────────────────────────────────────────────────────── */
void tm_run(TrainingManager *tm, long long steps)
{
    tm->total_steps    = steps;
    tm->counter        = 0;
    tm->graph_built    = 0;
    tm->frames_since_last = 0;

    /* Register signal handlers so Ctrl+C saves and exits cleanly */
    g_stop = 0;
    signal(SIGINT,  handle_stop);
    signal(SIGTERM, handle_stop);

    long long t0 = now_ns();
    tm->t_start_ns = t0;
    tm->t_last_ns  = t0;

    /* Prime data provider */
    tm->data->advance(tm->data);

    for (tm->counter = 0; tm->counter < steps && !g_stop; tm->counter++) {
        tm_step(tm);

        /* Build CUDA graph after warm-up */
        if (!tm->graph_built &&
            tm->opts.use_cuda_graph &&
            tm->counter >= tm->opts.warmup_steps)
        {
            CUDA_CHECK(cudaStreamSynchronize(tm->pvm->stream_compute));
            pvm_build_cuda_graph(tm->pvm);
            tm->graph_built = 1;
        }

        /* Print progress */
        print_progress(tm);

        /* Checkpoint save */
        if (tm->opts.save_every > 0 &&
            tm->pvm->step > 0 &&
            tm->pvm->step % tm->opts.save_every == 0)
        {
            char path[512];
            snprintf(path, sizeof(path), "%s_%s_%09lld.bin",
                     tm->opts.save_prefix,
                     tm->opts.model_name,
                     (long long)tm->pvm->step);
            pvm_save(tm->pvm, path);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(tm->pvm->stream_compute));
    printf("\n");

    if (g_stop) {
        printf("Interrupted — saving checkpoint...\n");
        char path[512];
        snprintf(path, sizeof(path), "%s_%s_%09lld.bin",
                 tm->opts.save_prefix, tm->opts.model_name,
                 (long long)tm->pvm->step);
        pvm_save(tm->pvm, path);
    }
}
