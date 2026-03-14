/* =========================================================================
 *  readout.cu  --  C/CUDA port of the Python Readout class
 *
 *  Provides a heatmap readout head that maps PVM representation activations
 *  through an MLPCollection to produce per-block spatial predictions.
 * ========================================================================= */

#include "readout.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define RDO_MAGIC "RDO1"

/* ── Helpers ────────────────────────────────────────────────────────────── */

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

static int *alloc_int(int n)
{
    int *p = (int *)calloc(n, sizeof(int));
    if (!p) { fprintf(stderr, "readout alloc_int: OOM\n"); exit(1); }
    return p;
}

static void upload_int(int **d_ptr, const int *h_ptr, int n)
{
    CUDA_CHECK(cudaMalloc((void **)d_ptr, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, h_ptr, n * sizeof(int),
                           cudaMemcpyHostToDevice));
}

/* Check whether val appears in arr[0..n-1]. */
static int contains(const int *arr, int n, int val)
{
    for (int i = 0; i < n; i++)
        if (arr[i] == val) return 1;
    return 0;
}

/* =========================================================================
 *  readout_create
 * ========================================================================= */

ReadoutObject *readout_create(PVMObject *pvm, int representation_size,
                               int heatmap_block_size)
{
    ReadoutObject *ro = (ReadoutObject *)calloc(1, sizeof(ReadoutObject));
    if (!ro) { fprintf(stderr, "readout_create: OOM\n"); return NULL; }

    ro->pvm           = pvm;
    ro->readout_layer = 2;
    ro->blocks_x      = pvm->config.layer_shapes[0];
    ro->blocks_y      = pvm->config.layer_shapes[0];

    if (heatmap_block_size <= 0)
        ro->heatmap_block_size = pvm->config.input_block_size;
    else
        ro->heatmap_block_size = heatmap_block_size;

    ro->shape       = ro->blocks_x * ro->heatmap_block_size;
    ro->total_units = ro->blocks_x * ro->blocks_y;
    ro->opt_abs_diff = pvm->config.opt_abs_diff;

    /* ------------------------------------------------------------------
     *  Pass 1 : count total copy entries (blocks) and per-unit input sizes
     * ------------------------------------------------------------------ */
    int *sizes = alloc_int(ro->total_units);
    int total_blocks = 0;
    int idx = 0;

    for (int x = 0; x < ro->blocks_x; x++) {
        for (int y = 0; y < ro->blocks_y; y++) {
            for (int b = 0; b < pvm->total_units; b++) {
                PVMBlock *blk = &pvm->graph[b];
                if (contains(blk->xs, blk->n_xs, x) &&
                    contains(blk->ys, blk->n_ys, y)) {
                    sizes[idx] += blk->size;
                    total_blocks++;
                }
            }
            idx++;
        }
    }

    ro->total_blocks = total_blocks;
    ro->h_sizes      = sizes;

    /* ------------------------------------------------------------------
     *  Pass 2 : build ptrs_from, ptrs_to, qnt_from and MLP specs
     * ------------------------------------------------------------------ */
    ro->h_ptrs_from = alloc_int(total_blocks);
    ro->h_ptrs_to   = alloc_int(total_blocks);
    ro->h_qnt_from  = alloc_int(total_blocks);

    int **mlp_specs   = (int **)malloc(ro->total_units * sizeof(int *));
    int  *spec_lengths = (int *)malloc(ro->total_units * sizeof(int));
    if (!mlp_specs || !spec_lengths) {
        fprintf(stderr, "readout_create: OOM (specs)\n"); exit(1);
    }

    idx = 0;
    int b_idx = 0;
    int running_ptr = 0;

    for (int x = 0; x < ro->blocks_x; x++) {
        for (int y = 0; y < ro->blocks_y; y++) {
            for (int b = 0; b < pvm->total_units; b++) {
                PVMBlock *blk = &pvm->graph[b];
                if (contains(blk->xs, blk->n_xs, x) &&
                    contains(blk->ys, blk->n_ys, y)) {
                    ro->h_ptrs_from[b_idx] = pvm->h_repr_ptr[blk->id];
                    ro->h_ptrs_to[b_idx]   = running_ptr;
                    ro->h_qnt_from[b_idx]  = blk->size;
                    running_ptr += blk->size;
                    b_idx++;
                }
            }
            running_ptr += 1;   /* skip the bias unit slot */

            /* MLP spec: [input_size, hidden_size, output_size] */
            int hbs2 = ro->heatmap_block_size * ro->heatmap_block_size;
            mlp_specs[idx]   = (int *)malloc(3 * sizeof(int));
            mlp_specs[idx][0] = sizes[idx];
            mlp_specs[idx][1] = representation_size;
            mlp_specs[idx][2] = hbs2;
            spec_lengths[idx] = 3;

            idx++;
        }
    }

    /* ------------------------------------------------------------------
     *  Create the MLP collection and upload to GPU
     * ------------------------------------------------------------------ */
    float lr  = 0.01f;
    float mom = pvm->config.momentum_val;

    ro->mlp = mlp_create(ro->total_units, mlp_specs, spec_lengths, lr, mom);
    mlp_generate_gpu_mem(ro->mlp);
    mlp_set_gpu_mem(ro->mlp);

    for (int j = 0; j < ro->total_units; j++)
        free(mlp_specs[j]);
    free(mlp_specs);
    free(spec_lengths);

    /* Upload copy-pointer arrays to device */
    upload_int(&ro->d_ptrs_from, ro->h_ptrs_from, total_blocks);
    upload_int(&ro->d_ptrs_to,   ro->h_ptrs_to,   total_blocks);
    upload_int(&ro->d_qnt_from,  ro->h_qnt_from,  total_blocks);

    printf("Readout created: %d units, %d copy blocks, shape %dx%d\n",
           ro->total_units, ro->total_blocks, ro->shape, ro->shape);
    return ro;
}

/* =========================================================================
 *  readout_copy_data  --  scatter PVM repr activations into MLP input layer
 * ========================================================================= */

void readout_copy_data(ReadoutObject *ro)
{
    int current_step = ro->pvm->step % PVM_SEQ_LENGTH;
    int tb = ro->total_blocks;

    int blk = (tb < 128) ? tb : 128;
    int grd = tb / 128 + 1;

    gpu_copy_blocks<<<dim3(grd, 1), dim3(blk, 1, 1)>>>(
        ro->pvm->d_repr_activation[current_step],
        ro->mlp->d_input_mem[0],
        ro->d_ptrs_from,
        ro->d_qnt_from,
        ro->d_ptrs_to,
        tb);
}

/* =========================================================================
 *  readout_forward  --  run the MLP forward pass
 * ========================================================================= */

void readout_forward(ReadoutObject *ro)
{
    mlp_forward(ro->mlp, 0);
}

/* =========================================================================
 *  readout_train  --  compute error from label and back-propagate
 * ========================================================================= */

void readout_train(ReadoutObject *ro, const float *label, int h, int w)
{
    /* Upload label to GPU */
    int label_size = h * w;
    float *d_label = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_label, label_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_label, label, label_size * sizeof(float),
                           cudaMemcpyHostToDevice));

    int rl = ro->readout_layer;
    int tu = ro->total_units;
    int blk = (tu < 128) ? tu : 128;
    int grd = tu / 128 + 1;

    if (ro->opt_abs_diff) {
        gpu_calc_abs_diff_error_frame_1ch<<<dim3(grd, 1), dim3(blk, 1, 1)>>>(
            d_label,
            ro->mlp->d_input_mem[rl],
            ro->mlp->d_input_ptr[rl],
            ro->mlp->d_error_mem[rl],
            ro->mlp->d_error_ptr[rl],
            h, w,
            ro->blocks_x, ro->blocks_y,
            ro->heatmap_block_size, ro->heatmap_block_size,
            0,
            tu);
    } else {
        gpu_calc_error_frame_1ch<<<dim3(grd, 1), dim3(blk, 1, 1)>>>(
            d_label,
            ro->mlp->d_input_mem[rl],
            ro->mlp->d_input_ptr[rl],
            ro->mlp->d_error_mem[rl],
            ro->mlp->d_error_ptr[rl],
            h, w,
            ro->blocks_x, ro->blocks_y,
            ro->heatmap_block_size, ro->heatmap_block_size,
            0,
            tu);
    }

    mlp_backward(ro->mlp, 0, 0, 0);

    CUDA_CHECK(cudaFree(d_label));
}

/* =========================================================================
 *  readout_get_heatmap  --  collect MLP output into a spatial frame
 * ========================================================================= */

void readout_get_heatmap(ReadoutObject *ro, float *out_buf)
{
    int rl = ro->readout_layer;
    int frame_size = ro->shape * ro->shape;

    float *d_frame = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_frame, frame_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_frame, 0, frame_size * sizeof(float)));

    int tu  = ro->total_units;
    int blk = (tu < 128) ? tu : 128;
    int grd = tu / 128 + 1;

    gpu_collect_frame_1ch<<<dim3(grd, 1), dim3(blk, 1, 1)>>>(
        d_frame,
        ro->mlp->d_input_mem[rl],
        ro->mlp->d_input_ptr[rl],
        ro->shape, ro->shape,
        ro->blocks_x, ro->blocks_y,
        ro->heatmap_block_size, ro->heatmap_block_size,
        0,
        tu);

    CUDA_CHECK(cudaMemcpy(out_buf, d_frame, frame_size * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_frame));
}

/* =========================================================================
 *  readout_update_learning_rate
 * ========================================================================= */

void readout_update_learning_rate(ReadoutObject *ro, float override_rate)
{
    float mul = 10.0f;
    float rate = -1.0f;

    if (ro->pvm->step == ro->pvm->config.delay_final_learning_rate) {
        rate = mul * ro->pvm->config.final_learning_rate;
        printf("Setting final readout learning rate: %f\n", rate);
    }

    if (ro->pvm->config.delay_intermediate_learning_rate > 0 &&
        ro->pvm->step == ro->pvm->config.delay_intermediate_learning_rate) {
        rate = mul * ro->pvm->config.intermediate_learning_rate;
        printf("Setting intermediate readout learning rate: %f\n", rate);
    }

    if (override_rate >= 0.0f) {
        rate = override_rate;
        printf("Overriding readout learning rate to %f\n", rate);
    }

    if (rate >= 0.0f)
        mlp_set_learning_rate(ro->mlp, rate);
}

/* =========================================================================
 *  readout_set_pvm  --  attach a (possibly new) PVM and set initial LR
 * ========================================================================= */

void readout_set_pvm(ReadoutObject *ro, PVMObject *pvm)
{
    ro->pvm = pvm;

    if (ro->heatmap_block_size <= 0)
        ro->heatmap_block_size = pvm->config.input_block_size;

    float mul  = 10.0f;
    float rate = mul * pvm->learning_rate;
    mlp_set_learning_rate(ro->mlp, rate);
}

/* =========================================================================
 *  readout_save  --  binary format with "RDO1" header
 *
 *  Layout:
 *    [4 bytes]  "RDO1"
 *    [int]      readout_layer
 *    [int]      heatmap_block_size
 *    [int]      blocks_x
 *    [int]      blocks_y
 *    [int]      shape
 *    [int]      total_units
 *    [int]      total_blocks
 *    [int]      opt_abs_diff
 *    [total_blocks * int]  h_ptrs_from
 *    [total_blocks * int]  h_ptrs_to
 *    [total_blocks * int]  h_qnt_from
 *    [total_units  * int]  h_sizes
 *
 *  The MLP is saved separately to <filename>.mlp
 * ========================================================================= */

int readout_save(ReadoutObject *ro, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) { perror("readout_save: fopen"); return -1; }

    fwrite(RDO_MAGIC, 1, 4, fp);
    fwrite(&ro->readout_layer,      sizeof(int), 1, fp);
    fwrite(&ro->heatmap_block_size, sizeof(int), 1, fp);
    fwrite(&ro->blocks_x,           sizeof(int), 1, fp);
    fwrite(&ro->blocks_y,           sizeof(int), 1, fp);
    fwrite(&ro->shape,              sizeof(int), 1, fp);
    fwrite(&ro->total_units,        sizeof(int), 1, fp);
    fwrite(&ro->total_blocks,       sizeof(int), 1, fp);
    fwrite(&ro->opt_abs_diff,       sizeof(int), 1, fp);

    fwrite(ro->h_ptrs_from, sizeof(int), ro->total_blocks, fp);
    fwrite(ro->h_ptrs_to,   sizeof(int), ro->total_blocks, fp);
    fwrite(ro->h_qnt_from,  sizeof(int), ro->total_blocks, fp);
    fwrite(ro->h_sizes,     sizeof(int), ro->total_units,  fp);

    fclose(fp);

    /* Save MLP to a companion file */
    size_t len = strlen(filename);
    char *mlp_path = (char *)malloc(len + 5);   /* ".mlp" + NUL */
    if (!mlp_path) { perror("readout_save: malloc"); return -1; }
    sprintf(mlp_path, "%s.mlp", filename);

    int rc = mlp_save(ro->mlp, mlp_path);
    free(mlp_path);

    return rc;
}

/* =========================================================================
 *  readout_load  --  read back from the binary format produced by save
 * ========================================================================= */

ReadoutObject *readout_load(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("readout_load: fopen"); return NULL; }

    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, RDO_MAGIC, 4) != 0) {
        fprintf(stderr, "readout_load: bad magic\n");
        fclose(fp);
        return NULL;
    }

    ReadoutObject *ro = (ReadoutObject *)calloc(1, sizeof(ReadoutObject));
    if (!ro) { fclose(fp); return NULL; }

    ro->pvm = NULL;   /* caller must attach via readout_set_pvm */

    fread(&ro->readout_layer,      sizeof(int), 1, fp);
    fread(&ro->heatmap_block_size, sizeof(int), 1, fp);
    fread(&ro->blocks_x,           sizeof(int), 1, fp);
    fread(&ro->blocks_y,           sizeof(int), 1, fp);
    fread(&ro->shape,              sizeof(int), 1, fp);
    fread(&ro->total_units,        sizeof(int), 1, fp);
    fread(&ro->total_blocks,       sizeof(int), 1, fp);
    fread(&ro->opt_abs_diff,       sizeof(int), 1, fp);

    ro->h_ptrs_from = alloc_int(ro->total_blocks);
    ro->h_ptrs_to   = alloc_int(ro->total_blocks);
    ro->h_qnt_from  = alloc_int(ro->total_blocks);
    ro->h_sizes     = alloc_int(ro->total_units);

    fread(ro->h_ptrs_from, sizeof(int), ro->total_blocks, fp);
    fread(ro->h_ptrs_to,   sizeof(int), ro->total_blocks, fp);
    fread(ro->h_qnt_from,  sizeof(int), ro->total_blocks, fp);
    fread(ro->h_sizes,     sizeof(int), ro->total_units,  fp);

    fclose(fp);

    /* Upload copy-pointer arrays to device */
    upload_int(&ro->d_ptrs_from, ro->h_ptrs_from, ro->total_blocks);
    upload_int(&ro->d_ptrs_to,   ro->h_ptrs_to,   ro->total_blocks);
    upload_int(&ro->d_qnt_from,  ro->h_qnt_from,  ro->total_blocks);

    /* Load MLP from companion file */
    size_t len = strlen(filename);
    char *mlp_path = (char *)malloc(len + 5);
    if (!mlp_path) {
        fprintf(stderr, "readout_load: OOM (mlp_path)\n");
        readout_destroy(ro);
        return NULL;
    }
    sprintf(mlp_path, "%s.mlp", filename);

    ro->mlp = mlp_load(mlp_path);
    free(mlp_path);

    if (!ro->mlp) {
        fprintf(stderr, "readout_load: failed to load MLP\n");
        readout_destroy(ro);
        return NULL;
    }

    printf("Readout loaded: %d units, %d copy blocks, shape %dx%d\n",
           ro->total_units, ro->total_blocks, ro->shape, ro->shape);
    return ro;
}

/* =========================================================================
 *  readout_destroy
 * ========================================================================= */

void readout_destroy(ReadoutObject *ro)
{
    if (!ro) return;

    /* Host arrays */
    free(ro->h_ptrs_from);
    free(ro->h_ptrs_to);
    free(ro->h_qnt_from);
    free(ro->h_sizes);

    /* Device arrays */
    if (ro->d_ptrs_from) cudaFree(ro->d_ptrs_from);
    if (ro->d_ptrs_to)   cudaFree(ro->d_ptrs_to);
    if (ro->d_qnt_from)  cudaFree(ro->d_qnt_from);

    /* MLP (owned) */
    if (ro->mlp) mlp_destroy(ro->mlp);

    free(ro);
}
