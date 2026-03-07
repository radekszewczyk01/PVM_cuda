/* infer_runner.c – Forward-only inference loop (plain C99)
 *
 * Runs the PVM forward pass on a data stream without weight updates.
 * Saves side-by-side PNG frames (left: input, right: prediction) to disk
 * and reports Mean Absolute Error statistics on stdout.
 */
#include "infer_runner.h"
#include "pvm_object.h"
#include "data_provider.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/* ── stb_image_write (single-header PNG/JPEG encoder) ─────────────────── */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* ── SIGINT/SIGTERM: allow clean early exit ──────────────────────────── */
static volatile sig_atomic_t g_infer_stop = 0;
static void handle_infer_stop(int sig) { (void)sig; g_infer_stop = 1; }

/* ── Monotonic nanosecond timer ─────────────────────────────────────────── */
static long long now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ── Helper: ensure directory exists ─────────────────────────────────── */
static int ensure_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0) return 0;   /* already exists */
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "[infer] Cannot create output dir '%s': %s\n",
                path, strerror(errno));
        return -1;
    }
    return 0;
}

/* ── Helper: clamp float to [0,1] and scale to uint8 ─────────────────── */
static inline uint8_t f2u8(float v)
{
    v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    return (uint8_t)(v * 255.f + 0.5f);
}

/* ── Default options ─────────────────────────────────────────────────── */
void infer_default_options(InferOptions *opts)
{
    memset(opts, 0, sizeof(*opts));
    strncpy(opts->out_dir, "/tmp/pvm_infer", sizeof(opts->out_dir) - 1);
    opts->steps      = 1000;
    opts->save_every = 10;
    opts->warmup     = 3;
}

/* ── run_infer ───────────────────────────────────────────────────────── */
double run_infer(void *pvm_vp, void *data_vp, const InferOptions *opts_in)
{
    PVMObject    *pvm  = (PVMObject *)pvm_vp;
    DataProvider *data = (DataProvider *)data_vp;

    InferOptions opts;
    if (opts_in) opts = *opts_in;
    else         infer_default_options(&opts);

    /* Geometry */
    int L0 = pvm->cfg.layer_shapes[0];
    int I  = pvm->cfg.input_block_size;
    int W  = L0 * I;
    int H  = W;
    int C  = pvm->cfg.input_channels;
    size_t frame_floats = (size_t)W * H * C;

    /* Allocate host buffers */
    float   *pred_host = (float   *)malloc(frame_floats * sizeof(float));
    float   *inp_copy  = (float   *)malloc(frame_floats * sizeof(float));
    /* Canvas for side-by-side PNG: 2*W wide, H tall, always 3 channels */
    uint8_t *canvas    = (uint8_t *)malloc((size_t)W * 2 * H * 3);
    if (!pred_host || !inp_copy || !canvas) {
        fprintf(stderr, "[infer] Out of memory\n");
        free(pred_host); free(inp_copy); free(canvas);
        return -1.0;
    }

    /* Create output directory */
    if (opts.save_every > 0)
        ensure_dir(opts.out_dir);

    /* Signal handlers so Ctrl+C finishes cleanly */
    g_infer_stop = 0;
    signal(SIGINT,  handle_infer_stop);
    signal(SIGTERM, handle_infer_stop);

    printf("[infer] Frame size: %dx%d  channels: %d\n", W, H, C);
    printf("[infer] Warm-up: %d steps  Run: %d steps  Save every: %d\n",
           opts.warmup, opts.steps, opts.save_every);
    if (opts.save_every > 0)
        printf("[infer] Output directory: %s\n", opts.out_dir);
    printf("[infer] Starting at model step %lld\n\n", (long long)pvm->step);
    fflush(stdout);

    double total_mae   = 0.0;
    int    data_frames = 0;    /* frames counted for MAE (excludes warm-up) */
    long long t_start  = now_ns();
    long long t_prev   = t_start;
    int total_run      = opts.warmup + (opts.steps > 0 ? opts.steps : 999999999);

    for (int s = 0; s < total_run && !g_infer_stop; s++) {
        /* Advance data source */
        data->advance(data);
        const float *fr = data->get_frame(data);
        if (!fr) {
            printf("\n[infer] Data source exhausted at step %d\n", s);
            break;
        }

        /* Copy to pinned staging buffer (batch_size forced to 1) */
        memcpy(pvm->pinned_frame, fr, frame_floats * sizeof(float));
        /* Keep a copy for MAE / PNG */
        memcpy(inp_copy, fr, frame_floats * sizeof(float));

        /* Forward pass – no backward, no weight update */
        pvm_push_input(pvm, pvm->pinned_frame);
        pvm_forward(pvm);

        /* Download prediction */
        pvm_pop_prediction(pvm, pred_host, /*batch=*/0);

        /* Manually advance step counter (normally done inside pvm_backward) */
        pvm->step++;

        /* Skip MAE / save during warm-up */
        if (s < opts.warmup) continue;

        /* MAE */
        double mae = 0.0;
        for (size_t i = 0; i < frame_floats; i++)
            mae += fabs(inp_copy[i] - pred_host[i]);
        mae /= (double)frame_floats;
        total_mae += mae;
        data_frames++;

        /* Progress line */
        long long now = now_ns();
        if ((now - t_prev) > 500000000LL || data_frames == 1) {  /* every 0.5 s */
            double elapsed = (double)(now - t_start) * 1e-9;
            double fps     = (elapsed > 0.0) ? (double)data_frames / elapsed : 0.0;
            printf("\r[infer] frame %5d  MAE: %.4f  avg MAE: %.4f  fps: %.1f    ",
                   data_frames,
                   (float)mae,
                   (float)(total_mae / data_frames),
                   fps);
            fflush(stdout);
            t_prev = now;
        }

        /* Save side-by-side PNG */
        if (opts.save_every > 0 && (data_frames % opts.save_every == 0)) {
            /* Build canvas: left = input, right = prediction */
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    for (int c = 0; c < 3; c++) {
                        /* Source channel index (handle 1-channel models) */
                        int src_c = (C >= 3) ? c : 0;
                        int fi    = (y * W + x) * C + src_c;
                        uint8_t ui = f2u8(inp_copy[fi]);
                        uint8_t up = f2u8(pred_host[fi]);
                        /* Left half: input */
                        canvas[(y * W * 2 + x) * 3 + c] = ui;
                        /* Right half: prediction */
                        canvas[(y * W * 2 + (x + W)) * 3 + c] = up;
                    }
                }
            }
            char fname[640];
            snprintf(fname, sizeof(fname), "%s/frame_%06d.png", opts.out_dir, data_frames);
            stbi_write_png(fname, W * 2, H, 3, canvas, W * 2 * 3);
        }
    }

    double avg_mae = (data_frames > 0) ? (total_mae / data_frames) : 0.0;
    double elapsed_s = (double)(now_ns() - t_start) * 1e-9;

    printf("\n\n=== Inference complete ===\n");
    printf("  Frames processed : %d\n", data_frames);
    printf("  Average MAE      : %.5f\n", (float)avg_mae);
    printf("  Elapsed time     : %.1f s\n", elapsed_s);
    if (elapsed_s > 0.0)
        printf("  Throughput       : %.1f fps\n",
               (double)data_frames / elapsed_s);
    if (opts.save_every > 0)
        printf("  PNG frames saved : %s/\n", opts.out_dir);
    fflush(stdout);

    free(pred_host);
    free(inp_copy);
    free(canvas);

    return avg_mae;
}
