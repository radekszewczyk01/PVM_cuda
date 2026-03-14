/* =========================================================================
 *  mlp.cu  --  C/CUDA port of the Python MLP_collection class
 *
 *  Manages a batch of identically-structured multi-layer perceptrons
 *  (up to MLP_MAX_LAYERS weight layers) on the GPU.
 * ========================================================================= */

#include "mlp.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Helpers */

static inline float randf(void) { return (float)rand() / (float)RAND_MAX; }

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

/* Helper: host-allocate and zero an int array. */
static int *alloc_int(int n)
{
    int *p = (int *)calloc(n, sizeof(int));
    if (!p) { fprintf(stderr, "alloc_int: OOM\n"); exit(1); }
    return p;
}

/* Helper: host-allocate and zero a float array. */
static float *alloc_float(int n)
{
    float *p = (float *)calloc(n, sizeof(float));
    if (!p) { fprintf(stderr, "alloc_float: OOM\n"); exit(1); }
    return p;
}

/* Helper: host-allocate a float array filled with a constant. */
static float *alloc_float_fill(int n, float val)
{
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "alloc_float_fill: OOM\n"); exit(1); }
    for (int k = 0; k < n; k++) p[k] = val;
    return p;
}

/* Helper: GPU-allocate nbytes, copy host data in. */
static void upload_float(float **d_ptr, const float *h_ptr, int n)
{
    CUDA_CHECK(cudaMalloc((void **)d_ptr, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, h_ptr, n * sizeof(float),
                           cudaMemcpyHostToDevice));
}

static void upload_int(int **d_ptr, const int *h_ptr, int n)
{
    CUDA_CHECK(cudaMalloc((void **)d_ptr, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, h_ptr, n * sizeof(int),
                           cudaMemcpyHostToDevice));
}

/* =========================================================================
 *  mlp_create
 * ========================================================================= */

MLPCollection *mlp_create(int n_specs, int **specs, int *spec_lengths,
                           float learning_rate, float momentum)
{
    MLPCollection *mlp = (MLPCollection *)calloc(1, sizeof(MLPCollection));
    if (!mlp) { fprintf(stderr, "mlp_create: OOM\n"); return NULL; }

    srand((unsigned)time(NULL));

    mlp->flip_buf = 0;
    mlp->propagate_all_the_way = 0;
    mlp->input_layer_req = 0;

    /* ------------------------------------------------------------------
     *  Pass 1 : compute layerwise memory requirements
     * ------------------------------------------------------------------ */
    for (int cl = 0; cl < MLP_MAX_LAYERS; cl++) {
        mlp->layerwise_weight_mem_req[cl] = 0;
        mlp->layerwise_weight_objects[cl] = 0;
        mlp->layerwise_output_mem_req[cl] = 0;
    }

    for (int cl = 0; cl < MLP_MAX_LAYERS; cl++) {
        for (int s = 0; s < n_specs; s++) {
            int slen = spec_lengths[s];
            for (int i = 0; i < slen; i++) {
                if (i == cl) {
                    if (i < slen - 1) {
                        mlp->layerwise_weight_mem_req[i] +=
                            (specs[s][i] + 1) * specs[s][i + 1];
                        mlp->layerwise_weight_objects[i] += 1;
                        mlp->layerwise_output_mem_req[i] +=
                            (specs[s][i + 1] + 1);   /* +1 for bias unit */
                    }
                    if (i == 0) {
                        mlp->input_layer_req += (specs[s][i] + 1);
                    }
                }
            }
        }
    }

    /* total_objects = max of layerwise_weight_objects */
    int max_obj = 0;
    for (int i = 0; i < MLP_MAX_LAYERS; i++)
        if (mlp->layerwise_weight_objects[i] > max_obj)
            max_obj = mlp->layerwise_weight_objects[i];
    mlp->total_objects = max_obj;

    /* CUDA launch heuristics (matches Python) */
    int thread_num;
    if (mlp->total_objects > 8000)       thread_num = 256;
    else if (mlp->total_objects > 3000)  thread_num = 196;
    else                                 thread_num = 128;
    mlp->threads = (mlp->total_objects < thread_num)
                        ? mlp->total_objects : thread_num;
    mlp->grid_size = mlp->total_objects / thread_num + 1;

    /* Determine max_layers */
    int max_layers = 0;
    for (int i = 0; i < MLP_MAX_LAYERS; i++) {
        if (mlp->layerwise_output_mem_req[i] == 0) break;
        max_layers++;
    }
    mlp->max_layers = max_layers;

    /* ------------------------------------------------------------------
     *  Pass 2 : allocate and initialise per-weight-layer arrays
     * ------------------------------------------------------------------ */
    for (int cl = 0; cl < max_layers; cl++) {
        int n_obj = mlp->layerwise_weight_objects[cl];
        int wmem  = mlp->layerwise_weight_mem_req[cl];
        int omem  = mlp->layerwise_output_mem_req[cl];

        /* Weight memory */
        mlp->h_weight_mem[cl]  = alloc_float(wmem);
        mlp->h_weight_buf[cl]  = alloc_float(wmem);
        mlp->h_dweight_mem[cl] = alloc_float(wmem);
        mlp->h_mweight_mem[cl] = alloc_float(wmem);

        /* Pointer / shape arrays (n_obj + 1 for sentinel) */
        mlp->h_weight_ptr[cl] = alloc_int(n_obj + 1);
        mlp->h_shape0[cl]     = alloc_int(n_obj);
        mlp->h_shape1[cl]     = alloc_int(n_obj);

        /* Per-object scalars */
        mlp->h_beta[cl]          = alloc_float_fill(n_obj, 1.0f);
        mlp->h_learning_rate[cl] = alloc_float_fill(n_obj, learning_rate);
        mlp->h_momentum[cl]      = alloc_float_fill(n_obj, momentum);

        /* Fill shape arrays */
        for (int j = 0; j < n_specs; j++) {
            if (cl < spec_lengths[j] - 1) {
                mlp->h_shape0[cl][j] = specs[j][cl + 1];
                mlp->h_shape1[cl][j] = specs[j][cl] + 1;   /* +1 bias */
            }
        }

        /* Build weight_ptr and initialise weights */
        mlp->h_weight_ptr[cl][0] = 0;
        for (int j = 0; j < n_obj; j++) {
            int s0 = mlp->h_shape0[cl][j];
            int s1 = mlp->h_shape1[cl][j];
            int base = mlp->h_weight_ptr[cl][j];
            float scale = 2.0f / sqrtf((float)s0 - 0.9f);
            for (int k = 0; k < s0 * s1; k++)
                mlp->h_weight_mem[cl][base + k] =
                    (randf() - 0.5f) * scale;
            mlp->h_weight_ptr[cl][j + 1] = base + s0 * s1;
        }

        /* Activation layer (index cl+1) : output, delta, error */
        int act = cl + 1;
        mlp->h_input_mem[act] = alloc_float(omem);
        mlp->h_input_ptr[act] = alloc_int(n_obj + 1);
        mlp->h_delta_mem[act] = alloc_float(omem);
        mlp->h_delta_ptr[act] = alloc_int(n_obj + 1);
        mlp->h_error_mem[act] = alloc_float(omem);
        mlp->h_error_ptr[act] = alloc_int(n_obj + 1);
        mlp->input_mem_size[act] = omem;

        mlp->h_input_ptr[act][0] = 0;
        mlp->h_delta_ptr[act][0] = 0;
        mlp->h_error_ptr[act][0] = 0;

        for (int j = 0; j < n_obj; j++) {
            int s0 = mlp->h_shape0[cl][j];
            int block = s0 + 1;   /* activation + bias */
            int base = mlp->h_input_ptr[act][j];

            /* set bias unit to 1.0 */
            mlp->h_input_mem[act][base + s0] = 1.0f;

            mlp->h_input_ptr[act][j + 1] = base + block;
            mlp->h_delta_ptr[act][j + 1] = mlp->h_delta_ptr[act][j] + block;
            mlp->h_error_ptr[act][j + 1] = mlp->h_error_ptr[act][j] + block;
        }

        /* Build obj_id / row_id for fast kernels (one thread per column) */
        int total_thr = 0;
        for (int j = 0; j < n_obj; j++)
            total_thr += mlp->h_shape1[cl][j];

        mlp->h_obj_id[cl] = alloc_int(total_thr);
        mlp->h_row_id[cl] = alloc_int(total_thr);
        int tid = 0;
        for (int j = 0; j < n_obj; j++) {
            for (int l = 0; l < mlp->h_shape1[cl][j]; l++) {
                mlp->h_obj_id[cl][tid] = j;
                mlp->h_row_id[cl][tid] = l;
                tid++;
            }
        }
        mlp->total_threads[cl] = total_thr;
        mlp->block_size[cl]    = 128;
        mlp->grid[cl]          = total_thr / 128 + 1;
    }

    /* ------------------------------------------------------------------
     *  Pass 3 : create input layer (activation layer 0)
     * ------------------------------------------------------------------ */
    {
        int ireq = mlp->input_layer_req;
        mlp->h_input_mem[0] = alloc_float(ireq);
        mlp->h_input_ptr[0] = alloc_int(n_specs + 1);
        mlp->h_delta_mem[0] = alloc_float(ireq);
        mlp->h_delta_ptr[0] = alloc_int(n_specs + 1);
        mlp->h_error_mem[0] = alloc_float(ireq);
        mlp->h_error_ptr[0] = alloc_int(n_specs + 1);
        mlp->input_mem_size[0] = ireq;

        mlp->h_input_ptr[0][0] = 0;
        mlp->h_delta_ptr[0][0] = 0;
        mlp->h_error_ptr[0][0] = 0;

        for (int j = 0; j < n_specs; j++) {
            int s1 = mlp->h_shape1[0][j];          /* input_dim + 1 */
            int base = mlp->h_input_ptr[0][j];
            mlp->h_input_mem[0][base + s1 - 1] = 1.0f;   /* bias */
            mlp->h_input_ptr[0][j + 1] = base + s1;
            mlp->h_delta_ptr[0][j + 1] = mlp->h_delta_ptr[0][j] + s1;
            mlp->h_error_ptr[0][j + 1] = mlp->h_error_ptr[0][j] + s1;
        }
    }

    return mlp;
}

/* =========================================================================
 *  mlp_destroy
 * ========================================================================= */

void mlp_destroy(MLPCollection *mlp)
{
    if (!mlp) return;

    for (int i = 0; i < mlp->max_layers; i++) {
        free(mlp->h_weight_mem[i]);
        free(mlp->h_weight_buf[i]);
        free(mlp->h_dweight_mem[i]);
        free(mlp->h_mweight_mem[i]);
        free(mlp->h_weight_ptr[i]);
        free(mlp->h_shape0[i]);
        free(mlp->h_shape1[i]);
        free(mlp->h_beta[i]);
        free(mlp->h_learning_rate[i]);
        free(mlp->h_momentum[i]);
        free(mlp->h_obj_id[i]);
        free(mlp->h_row_id[i]);

        if (mlp->d_weight_mem[i])  cudaFree(mlp->d_weight_mem[i]);
        if (mlp->d_weight_buf[i])  cudaFree(mlp->d_weight_buf[i]);
        if (mlp->d_dweight_mem[i]) cudaFree(mlp->d_dweight_mem[i]);
        if (mlp->d_mweight_mem[i]) cudaFree(mlp->d_mweight_mem[i]);
        if (mlp->d_weight_ptr[i])  cudaFree(mlp->d_weight_ptr[i]);
        if (mlp->d_shape0[i])      cudaFree(mlp->d_shape0[i]);
        if (mlp->d_shape1[i])      cudaFree(mlp->d_shape1[i]);
        if (mlp->d_beta[i])        cudaFree(mlp->d_beta[i]);
        if (mlp->d_learning_rate[i]) cudaFree(mlp->d_learning_rate[i]);
        if (mlp->d_momentum[i])    cudaFree(mlp->d_momentum[i]);
        if (mlp->d_obj_id[i])      cudaFree(mlp->d_obj_id[i]);
        if (mlp->d_row_id[i])      cudaFree(mlp->d_row_id[i]);
    }

    for (int i = 0; i <= mlp->max_layers; i++) {
        free(mlp->h_input_mem[i]);
        free(mlp->h_input_ptr[i]);
        free(mlp->h_delta_mem[i]);
        free(mlp->h_delta_ptr[i]);
        free(mlp->h_error_mem[i]);
        free(mlp->h_error_ptr[i]);

        if (mlp->d_input_mem[i]) cudaFree(mlp->d_input_mem[i]);
        if (mlp->d_input_ptr[i]) cudaFree(mlp->d_input_ptr[i]);
        if (mlp->d_delta_mem[i]) cudaFree(mlp->d_delta_mem[i]);
        if (mlp->d_delta_ptr[i]) cudaFree(mlp->d_delta_ptr[i]);
        if (mlp->d_error_mem[i]) cudaFree(mlp->d_error_mem[i]);
        if (mlp->d_error_ptr[i]) cudaFree(mlp->d_error_ptr[i]);
    }

    free(mlp);
}

/* =========================================================================
 *  mlp_generate_gpu_mem  --  upload all host arrays to the GPU
 * ========================================================================= */

void mlp_generate_gpu_mem(MLPCollection *mlp)
{
    int ml = mlp->max_layers;

    for (int i = 0; i < ml; i++) {
        int n_obj = mlp->layerwise_weight_objects[i];
        int wmem  = mlp->layerwise_weight_mem_req[i];

        upload_float(&mlp->d_weight_mem[i],  mlp->h_weight_mem[i],  wmem);
        upload_float(&mlp->d_weight_buf[i],  mlp->h_weight_buf[i],  wmem);
        upload_float(&mlp->d_dweight_mem[i], mlp->h_dweight_mem[i], wmem);
        upload_float(&mlp->d_mweight_mem[i], mlp->h_mweight_mem[i], wmem);
        upload_int(&mlp->d_weight_ptr[i], mlp->h_weight_ptr[i], n_obj + 1);
        upload_int(&mlp->d_shape0[i],     mlp->h_shape0[i],     n_obj);
        upload_int(&mlp->d_shape1[i],     mlp->h_shape1[i],     n_obj);
        upload_float(&mlp->d_beta[i],          mlp->h_beta[i],          n_obj);
        upload_float(&mlp->d_learning_rate[i], mlp->h_learning_rate[i], n_obj);
        upload_float(&mlp->d_momentum[i],      mlp->h_momentum[i],      n_obj);
        upload_int(&mlp->d_obj_id[i], mlp->h_obj_id[i], mlp->total_threads[i]);
        upload_int(&mlp->d_row_id[i], mlp->h_row_id[i], mlp->total_threads[i]);
    }

    /* Activation layers 0 .. max_layers */
    for (int i = 0; i <= ml; i++) {
        int sz = mlp->input_mem_size[i];
        /* Determine pointer array length.  For activation layer 0 and
         * layer max_layers the ptr array was allocated with n_specs+1
         * entries; for intermediate layers it is layerwise_weight_objects+1.
         * We can safely figure out the count from the ptr array sentinel
         * stored at the last written index.  However we need the count,
         * so compute it from the struct. */
        int ptr_n;
        if (i == 0 || i == ml)
            ptr_n = mlp->total_objects + 1;
        else
            ptr_n = mlp->layerwise_weight_objects[i - 1] + 1;

        upload_float(&mlp->d_input_mem[i], mlp->h_input_mem[i], sz);
        upload_int(&mlp->d_input_ptr[i],   mlp->h_input_ptr[i], ptr_n);
        upload_float(&mlp->d_delta_mem[i], mlp->h_delta_mem[i], sz);
        upload_int(&mlp->d_delta_ptr[i],   mlp->h_delta_ptr[i], ptr_n);
        upload_float(&mlp->d_error_mem[i], mlp->h_error_mem[i], sz);
        upload_int(&mlp->d_error_ptr[i],   mlp->h_error_ptr[i], ptr_n);
    }
}

/* =========================================================================
 *  mlp_set_gpu_mem  --  push all host arrays to existing GPU allocations
 * ========================================================================= */

void mlp_set_gpu_mem(MLPCollection *mlp)
{
    int ml = mlp->max_layers;

    for (int i = 0; i < ml; i++) {
        int n_obj = mlp->layerwise_weight_objects[i];
        int wmem  = mlp->layerwise_weight_mem_req[i];

        CUDA_CHECK(cudaMemcpy(mlp->d_weight_mem[i],  mlp->h_weight_mem[i],
                              wmem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_weight_buf[i],  mlp->h_weight_buf[i],
                              wmem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_dweight_mem[i], mlp->h_dweight_mem[i],
                              wmem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_mweight_mem[i], mlp->h_mweight_mem[i],
                              wmem * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_weight_ptr[i],  mlp->h_weight_ptr[i],
                              (n_obj + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_shape0[i], mlp->h_shape0[i],
                              n_obj * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_shape1[i], mlp->h_shape1[i],
                              n_obj * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_beta[i], mlp->h_beta[i],
                              n_obj * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_learning_rate[i], mlp->h_learning_rate[i],
                              n_obj * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_momentum[i], mlp->h_momentum[i],
                              n_obj * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_obj_id[i], mlp->h_obj_id[i],
                              mlp->total_threads[i] * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_row_id[i], mlp->h_row_id[i],
                              mlp->total_threads[i] * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    for (int i = 0; i <= ml; i++) {
        int sz = mlp->input_mem_size[i];
        int ptr_n;
        if (i == 0 || i == ml)
            ptr_n = mlp->total_objects + 1;
        else
            ptr_n = mlp->layerwise_weight_objects[i - 1] + 1;

        CUDA_CHECK(cudaMemcpy(mlp->d_input_mem[i], mlp->h_input_mem[i],
                              sz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_input_ptr[i], mlp->h_input_ptr[i],
                              ptr_n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_delta_mem[i], mlp->h_delta_mem[i],
                              sz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_delta_ptr[i], mlp->h_delta_ptr[i],
                              ptr_n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_error_mem[i], mlp->h_error_mem[i],
                              sz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_error_ptr[i], mlp->h_error_ptr[i],
                              ptr_n * sizeof(int), cudaMemcpyHostToDevice));
    }
}

/* =========================================================================
 *  mlp_get_gpu_mem  --  pull all GPU arrays back to host
 * ========================================================================= */

void mlp_get_gpu_mem(MLPCollection *mlp)
{
    int ml = mlp->max_layers;

    for (int i = 0; i < ml; i++) {
        int n_obj = mlp->layerwise_weight_objects[i];
        int wmem  = mlp->layerwise_weight_mem_req[i];

        CUDA_CHECK(cudaMemcpy(mlp->h_weight_mem[i],  mlp->d_weight_mem[i],
                              wmem * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_weight_buf[i],  mlp->d_weight_buf[i],
                              wmem * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_dweight_mem[i], mlp->d_dweight_mem[i],
                              wmem * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_mweight_mem[i], mlp->d_mweight_mem[i],
                              wmem * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_weight_ptr[i],  mlp->d_weight_ptr[i],
                              (n_obj + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_shape0[i], mlp->d_shape0[i],
                              n_obj * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_shape1[i], mlp->d_shape1[i],
                              n_obj * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_beta[i], mlp->d_beta[i],
                              n_obj * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_learning_rate[i], mlp->d_learning_rate[i],
                              n_obj * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_momentum[i], mlp->d_momentum[i],
                              n_obj * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_obj_id[i], mlp->d_obj_id[i],
                              mlp->total_threads[i] * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_row_id[i], mlp->d_row_id[i],
                              mlp->total_threads[i] * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i <= ml; i++) {
        int sz = mlp->input_mem_size[i];
        int ptr_n;
        if (i == 0 || i == ml)
            ptr_n = mlp->total_objects + 1;
        else
            ptr_n = mlp->layerwise_weight_objects[i - 1] + 1;

        CUDA_CHECK(cudaMemcpy(mlp->h_input_mem[i], mlp->d_input_mem[i],
                              sz * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_input_ptr[i], mlp->d_input_ptr[i],
                              ptr_n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_delta_mem[i], mlp->d_delta_mem[i],
                              sz * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_delta_ptr[i], mlp->d_delta_ptr[i],
                              ptr_n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_error_mem[i], mlp->d_error_mem[i],
                              sz * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_error_ptr[i], mlp->d_error_ptr[i],
                              ptr_n * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

/* =========================================================================
 *  mlp_set_input  --  copy flat data into the input activation layer (host)
 *                     and push to GPU
 * ========================================================================= */

void mlp_set_input(MLPCollection *mlp, const float *data)
{
    int sz = mlp->input_mem_size[0];
    memcpy(mlp->h_input_mem[0], data, sz * sizeof(float));
    CUDA_CHECK(cudaMemcpy(mlp->d_input_mem[0], mlp->h_input_mem[0],
                          sz * sizeof(float), cudaMemcpyHostToDevice));
}

/* =========================================================================
 *  mlp_forward
 * ========================================================================= */

void mlp_forward(MLPCollection *mlp, int poly)
{
    int ml = mlp->max_layers;

    for (int i = 0; i < ml; i++) {
        /* Zero the output activation for this layer */
        int out_sz = mlp->input_mem_size[i + 1];
        CUDA_CHECK(cudaMemset(mlp->d_input_mem[i + 1], 0,
                              out_sz * sizeof(float)));

        /* Matrix-vector product with bias set */
        gpu_dot_fast_set_bias<<<mlp->grid[i], mlp->block_size[i]>>>(
            mlp->d_weight_mem[i],
            mlp->d_input_mem[i],
            mlp->d_input_mem[i + 1],
            mlp->d_weight_ptr[i],
            mlp->d_input_ptr[i],
            mlp->d_input_ptr[i + 1],
            mlp->d_shape0[i],
            mlp->d_shape1[i],
            mlp->d_obj_id[i],
            mlp->d_row_id[i],
            mlp->total_threads[i]);

        /* Activation function */
        int sig_grid = mlp->grid_size;
        int sig_blk  = 512;
        if (!poly) {
            gpu_sigmoid_fast<<<sig_grid, sig_blk>>>(
                mlp->d_input_mem[i + 1],
                mlp->d_input_ptr[i + 1],
                mlp->d_beta[i],
                mlp->d_shape0[i],
                mlp->total_objects);
        } else {
            gpu_sigmoid_poly_fast<<<sig_grid, sig_blk>>>(
                mlp->d_input_mem[i + 1],
                mlp->d_input_ptr[i + 1],
                mlp->d_beta[i],
                mlp->d_shape0[i],
                mlp->total_objects);
        }
    }

    /* Pull final-layer activations, deltas and errors back to host */
    {
        int sz = mlp->input_mem_size[ml];
        CUDA_CHECK(cudaMemcpy(mlp->h_input_mem[ml], mlp->d_input_mem[ml],
                              sz * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_delta_mem[ml], mlp->d_delta_mem[ml],
                              sz * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mlp->h_error_mem[ml], mlp->d_error_mem[ml],
                              sz * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

/* =========================================================================
 *  mlp_backward
 * ========================================================================= */

void mlp_backward(MLPCollection *mlp, int poly,
                  int propagate_all_way, int propagate_only)
{
    int ml  = mlp->max_layers;
    int thr = mlp->threads;
    int grd = mlp->grid_size;

    for (int i = ml; i >= 0; i--) {
        /* ---- sigmoid derivative * error -> delta ---- */

        /* The Python code uses gpu_shape1 from the *activation* layer as
         * the "shape0" argument to sigmoid_der_mul.  For layer 0..ml-1
         * gpu_shape1 lives in layerwise_objects[i] and equals shape1 of
         * the weight layer.  For layer ml, the Python code creates
         * gpu_shape1 from shape0[ml-1].  We replicate that by choosing
         * the correct device pointer. */
        int *d_sh;
        if (i < ml)
            d_sh = mlp->d_shape1[i];
        else
            d_sh = mlp->d_shape0[ml - 1];

        if (!poly) {
            gpu_sigmoid_der_mul<<<grd, thr>>>(
                mlp->d_input_mem[i],
                mlp->d_error_mem[i],
                mlp->d_delta_mem[i],
                mlp->d_input_ptr[i],
                mlp->d_error_ptr[i],
                mlp->d_delta_ptr[i],
                d_sh,
                mlp->total_objects);
        } else {
            gpu_sigmoid_poly_der_mul<<<grd, thr>>>(
                mlp->d_input_mem[i],
                mlp->d_error_mem[i],
                mlp->d_delta_mem[i],
                mlp->d_input_ptr[i],
                mlp->d_error_ptr[i],
                mlp->d_delta_ptr[i],
                d_sh,
                mlp->total_objects);
        }

        if (i > 0) {
            /* ---- error back-propagation through weights ---- */
            int do_backprop = (i > 1) ||
                              (i == 1 && (mlp->propagate_all_the_way ||
                                          propagate_all_way));
            if (do_backprop) {
                int err_sz = mlp->input_mem_size[i - 1];
                CUDA_CHECK(cudaMemset(mlp->d_error_mem[i - 1], 0,
                                      err_sz * sizeof(float)));

                gpu_dot_transpose_fast<<<mlp->grid[i - 1],
                                         mlp->block_size[i - 1]>>>(
                    mlp->d_weight_mem[i - 1],
                    mlp->d_weight_buf[i - 1],
                    mlp->d_delta_mem[i],
                    mlp->d_weight_ptr[i - 1],
                    mlp->d_delta_ptr[i],
                    mlp->d_shape0[i - 1],
                    mlp->d_shape1[i - 1],
                    mlp->d_obj_id[i - 1],
                    mlp->d_row_id[i - 1],
                    mlp->total_threads[i - 1]);

                gpu_sum_dot_transpose<<<mlp->grid[i - 1],
                                        mlp->block_size[i - 1]>>>(
                    mlp->d_weight_buf[i - 1],
                    mlp->d_error_mem[i - 1],
                    mlp->d_weight_ptr[i - 1],
                    mlp->d_error_ptr[i - 1],
                    mlp->d_shape0[i - 1],
                    mlp->d_shape1[i - 1],
                    mlp->d_obj_id[i - 1],
                    mlp->d_row_id[i - 1],
                    mlp->total_threads[i - 1]);
            }

            /* ---- weight update via generalized outer product ---- */
            if (!mlp->flip_buf) {
                /* momentum in mweight, result in dweight */
                gpu_generalized_outer_fast<<<mlp->grid[i - 1],
                                             mlp->block_size[i - 1]>>>(
                    mlp->d_delta_mem[i],
                    mlp->d_input_mem[i - 1],
                    mlp->d_mweight_mem[i - 1],
                    mlp->d_dweight_mem[i - 1],
                    mlp->d_delta_ptr[i],
                    mlp->d_input_ptr[i - 1],
                    mlp->d_weight_ptr[i - 1],
                    mlp->d_weight_ptr[i - 1],
                    mlp->d_shape0[i - 1],
                    mlp->d_shape1[i - 1],
                    mlp->d_learning_rate[i - 1],
                    mlp->d_momentum[i - 1],
                    mlp->d_obj_id[i - 1],
                    mlp->d_row_id[i - 1],
                    mlp->total_threads[i - 1]);

                if (!propagate_only) {
                    /* d_weight_mem += d_dweight_mem  (device-to-device) */
                    int wmem = mlp->layerwise_weight_mem_req[i - 1];
                    float alpha = 1.0f;
                    /* Use a simple kernel-less approach via cublas would be
                     * ideal, but to keep dependency-free we launch gpu_add
                     * with total_obj = layerwise_weight_objects and treat
                     * each object as shape (shape0, shape1). */
                    int n_obj = mlp->layerwise_weight_objects[i - 1];
                    int add_grid = n_obj / 128 + 1;
                    gpu_add<<<add_grid, 128>>>(
                        mlp->d_dweight_mem[i - 1],
                        mlp->d_weight_mem[i - 1],
                        mlp->d_weight_ptr[i - 1],
                        mlp->d_weight_ptr[i - 1],
                        mlp->d_shape0[i - 1],
                        mlp->d_shape1[i - 1],
                        n_obj);
                }
            } else {
                /* momentum in dweight, result in mweight */
                gpu_generalized_outer_fast<<<mlp->grid[i - 1],
                                             mlp->block_size[i - 1]>>>(
                    mlp->d_delta_mem[i],
                    mlp->d_input_mem[i - 1],
                    mlp->d_dweight_mem[i - 1],
                    mlp->d_mweight_mem[i - 1],
                    mlp->d_delta_ptr[i],
                    mlp->d_input_ptr[i - 1],
                    mlp->d_weight_ptr[i - 1],
                    mlp->d_weight_ptr[i - 1],
                    mlp->d_shape0[i - 1],
                    mlp->d_shape1[i - 1],
                    mlp->d_learning_rate[i - 1],
                    mlp->d_momentum[i - 1],
                    mlp->d_obj_id[i - 1],
                    mlp->d_row_id[i - 1],
                    mlp->total_threads[i - 1]);

                if (!propagate_only) {
                    int n_obj = mlp->layerwise_weight_objects[i - 1];
                    int add_grid = n_obj / 128 + 1;
                    gpu_add<<<add_grid, 128>>>(
                        mlp->d_mweight_mem[i - 1],
                        mlp->d_weight_mem[i - 1],
                        mlp->d_weight_ptr[i - 1],
                        mlp->d_weight_ptr[i - 1],
                        mlp->d_shape0[i - 1],
                        mlp->d_shape1[i - 1],
                        n_obj);
                }
            }
        }
    }

    mlp->flip_buf = !mlp->flip_buf;
}

/* =========================================================================
 *  mlp_set_learning_rate
 * ========================================================================= */

void mlp_set_learning_rate(MLPCollection *mlp, float rate)
{
    printf("Setting MLP learning rate to %f\n", rate);
    for (int i = 0; i < mlp->max_layers; i++) {
        int n_obj = mlp->layerwise_weight_objects[i];
        for (int j = 0; j < n_obj; j++)
            mlp->h_learning_rate[i][j] = rate;
        CUDA_CHECK(cudaMemcpy(mlp->d_learning_rate[i],
                              mlp->h_learning_rate[i],
                              n_obj * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

/* =========================================================================
 *  mlp_save  --  binary format with "MLP1" header
 *
 *  Layout:
 *    [4 bytes]  "MLP1"
 *    [int]      max_layers
 *    [int]      total_objects
 *    [int]      flip_buf
 *    [int]      propagate_all_the_way
 *    [int]      input_layer_req
 *    [int]      threads
 *    [int]      grid_size
 *    [10*int]   layerwise_weight_mem_req
 *    [10*int]   layerwise_weight_objects
 *    [10*int]   layerwise_output_mem_req
 *    [11*int]   input_mem_size
 *    For each weight layer 0..max_layers-1:
 *      [int] total_threads[i], block_size[i], grid[i]
 *      [int] n_obj  (= layerwise_weight_objects[i])
 *      [int] wmem   (= layerwise_weight_mem_req[i])
 *      weight_mem, dweight_mem, mweight_mem, weight_buf  (wmem floats each)
 *      weight_ptr  (n_obj+1 ints)
 *      shape0, shape1  (n_obj ints each)
 *      beta, learning_rate, momentum  (n_obj floats each)
 *      obj_id, row_id  (total_threads[i] ints each)
 *    For each activation layer 0..max_layers:
 *      [int] sz (= input_mem_size[i])
 *      [int] ptr_n
 *      input_mem, delta_mem, error_mem  (sz floats each)
 *      input_ptr, delta_ptr, error_ptr  (ptr_n ints each)
 * ========================================================================= */

int mlp_save(MLPCollection *mlp, const char *filename)
{
    /* Pull latest state from GPU */
    mlp_get_gpu_mem(mlp);

    FILE *fp = fopen(filename, "wb");
    if (!fp) { perror("mlp_save: fopen"); return -1; }

    fwrite(MLP_MAGIC, 1, 4, fp);
    fwrite(&mlp->max_layers,           sizeof(int), 1, fp);
    fwrite(&mlp->total_objects,        sizeof(int), 1, fp);
    fwrite(&mlp->flip_buf,             sizeof(int), 1, fp);
    fwrite(&mlp->propagate_all_the_way,sizeof(int), 1, fp);
    fwrite(&mlp->input_layer_req,      sizeof(int), 1, fp);
    fwrite(&mlp->threads,              sizeof(int), 1, fp);
    fwrite(&mlp->grid_size,            sizeof(int), 1, fp);
    fwrite(mlp->layerwise_weight_mem_req,  sizeof(int), MLP_MAX_LAYERS, fp);
    fwrite(mlp->layerwise_weight_objects,  sizeof(int), MLP_MAX_LAYERS, fp);
    fwrite(mlp->layerwise_output_mem_req,  sizeof(int), MLP_MAX_LAYERS, fp);
    fwrite(mlp->input_mem_size,            sizeof(int), MLP_MAX_ACTIV,  fp);

    int ml = mlp->max_layers;
    for (int i = 0; i < ml; i++) {
        int n_obj = mlp->layerwise_weight_objects[i];
        int wmem  = mlp->layerwise_weight_mem_req[i];

        fwrite(&mlp->total_threads[i], sizeof(int), 1, fp);
        fwrite(&mlp->block_size[i],    sizeof(int), 1, fp);
        fwrite(&mlp->grid[i],          sizeof(int), 1, fp);
        fwrite(&n_obj,                 sizeof(int), 1, fp);
        fwrite(&wmem,                  sizeof(int), 1, fp);

        fwrite(mlp->h_weight_mem[i],  sizeof(float), wmem, fp);
        fwrite(mlp->h_dweight_mem[i], sizeof(float), wmem, fp);
        fwrite(mlp->h_mweight_mem[i], sizeof(float), wmem, fp);
        fwrite(mlp->h_weight_buf[i],  sizeof(float), wmem, fp);
        fwrite(mlp->h_weight_ptr[i],  sizeof(int),   n_obj + 1, fp);
        fwrite(mlp->h_shape0[i],      sizeof(int),   n_obj, fp);
        fwrite(mlp->h_shape1[i],      sizeof(int),   n_obj, fp);
        fwrite(mlp->h_beta[i],          sizeof(float), n_obj, fp);
        fwrite(mlp->h_learning_rate[i], sizeof(float), n_obj, fp);
        fwrite(mlp->h_momentum[i],      sizeof(float), n_obj, fp);
        fwrite(mlp->h_obj_id[i], sizeof(int), mlp->total_threads[i], fp);
        fwrite(mlp->h_row_id[i], sizeof(int), mlp->total_threads[i], fp);
    }

    for (int i = 0; i <= ml; i++) {
        int sz = mlp->input_mem_size[i];
        int ptr_n;
        if (i == 0 || i == ml)
            ptr_n = mlp->total_objects + 1;
        else
            ptr_n = mlp->layerwise_weight_objects[i - 1] + 1;

        fwrite(&sz,    sizeof(int), 1, fp);
        fwrite(&ptr_n, sizeof(int), 1, fp);
        fwrite(mlp->h_input_mem[i], sizeof(float), sz, fp);
        fwrite(mlp->h_delta_mem[i], sizeof(float), sz, fp);
        fwrite(mlp->h_error_mem[i], sizeof(float), sz, fp);
        fwrite(mlp->h_input_ptr[i], sizeof(int),   ptr_n, fp);
        fwrite(mlp->h_delta_ptr[i], sizeof(int),   ptr_n, fp);
        fwrite(mlp->h_error_ptr[i], sizeof(int),   ptr_n, fp);
    }

    fclose(fp);
    return 0;
}

/* =========================================================================
 *  mlp_load  --  read back from the binary format produced by mlp_save
 * ========================================================================= */

MLPCollection *mlp_load(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("mlp_load: fopen"); return NULL; }

    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, MLP_MAGIC, 4) != 0) {
        fprintf(stderr, "mlp_load: bad magic\n");
        fclose(fp);
        return NULL;
    }

    MLPCollection *mlp = (MLPCollection *)calloc(1, sizeof(MLPCollection));
    if (!mlp) { fclose(fp); return NULL; }

    fread(&mlp->max_layers,            sizeof(int), 1, fp);
    fread(&mlp->total_objects,         sizeof(int), 1, fp);
    fread(&mlp->flip_buf,              sizeof(int), 1, fp);
    fread(&mlp->propagate_all_the_way, sizeof(int), 1, fp);
    fread(&mlp->input_layer_req,       sizeof(int), 1, fp);
    fread(&mlp->threads,               sizeof(int), 1, fp);
    fread(&mlp->grid_size,             sizeof(int), 1, fp);
    fread(mlp->layerwise_weight_mem_req,  sizeof(int), MLP_MAX_LAYERS, fp);
    fread(mlp->layerwise_weight_objects,  sizeof(int), MLP_MAX_LAYERS, fp);
    fread(mlp->layerwise_output_mem_req,  sizeof(int), MLP_MAX_LAYERS, fp);
    fread(mlp->input_mem_size,            sizeof(int), MLP_MAX_ACTIV,  fp);

    int ml = mlp->max_layers;

    for (int i = 0; i < ml; i++) {
        int n_obj, wmem;
        fread(&mlp->total_threads[i], sizeof(int), 1, fp);
        fread(&mlp->block_size[i],    sizeof(int), 1, fp);
        fread(&mlp->grid[i],          sizeof(int), 1, fp);
        fread(&n_obj,                 sizeof(int), 1, fp);
        fread(&wmem,                  sizeof(int), 1, fp);

        mlp->h_weight_mem[i]  = alloc_float(wmem);
        mlp->h_dweight_mem[i] = alloc_float(wmem);
        mlp->h_mweight_mem[i] = alloc_float(wmem);
        mlp->h_weight_buf[i]  = alloc_float(wmem);
        fread(mlp->h_weight_mem[i],  sizeof(float), wmem, fp);
        fread(mlp->h_dweight_mem[i], sizeof(float), wmem, fp);
        fread(mlp->h_mweight_mem[i], sizeof(float), wmem, fp);
        fread(mlp->h_weight_buf[i],  sizeof(float), wmem, fp);

        mlp->h_weight_ptr[i] = alloc_int(n_obj + 1);
        fread(mlp->h_weight_ptr[i], sizeof(int), n_obj + 1, fp);

        mlp->h_shape0[i] = alloc_int(n_obj);
        mlp->h_shape1[i] = alloc_int(n_obj);
        fread(mlp->h_shape0[i], sizeof(int), n_obj, fp);
        fread(mlp->h_shape1[i], sizeof(int), n_obj, fp);

        mlp->h_beta[i]          = alloc_float(n_obj);
        mlp->h_learning_rate[i] = alloc_float(n_obj);
        mlp->h_momentum[i]      = alloc_float(n_obj);
        fread(mlp->h_beta[i],          sizeof(float), n_obj, fp);
        fread(mlp->h_learning_rate[i], sizeof(float), n_obj, fp);
        fread(mlp->h_momentum[i],      sizeof(float), n_obj, fp);

        int tt = mlp->total_threads[i];
        mlp->h_obj_id[i] = alloc_int(tt);
        mlp->h_row_id[i] = alloc_int(tt);
        fread(mlp->h_obj_id[i], sizeof(int), tt, fp);
        fread(mlp->h_row_id[i], sizeof(int), tt, fp);
    }

    for (int i = 0; i <= ml; i++) {
        int sz, ptr_n;
        fread(&sz,    sizeof(int), 1, fp);
        fread(&ptr_n, sizeof(int), 1, fp);

        mlp->h_input_mem[i] = alloc_float(sz);
        mlp->h_delta_mem[i] = alloc_float(sz);
        mlp->h_error_mem[i] = alloc_float(sz);
        fread(mlp->h_input_mem[i], sizeof(float), sz, fp);
        fread(mlp->h_delta_mem[i], sizeof(float), sz, fp);
        fread(mlp->h_error_mem[i], sizeof(float), sz, fp);

        mlp->h_input_ptr[i] = alloc_int(ptr_n);
        mlp->h_delta_ptr[i] = alloc_int(ptr_n);
        mlp->h_error_ptr[i] = alloc_int(ptr_n);
        fread(mlp->h_input_ptr[i], sizeof(int), ptr_n, fp);
        fread(mlp->h_delta_ptr[i], sizeof(int), ptr_n, fp);
        fread(mlp->h_error_ptr[i], sizeof(int), ptr_n, fp);
    }

    fclose(fp);

    /* Upload everything to GPU */
    mlp_generate_gpu_mem(mlp);

    return mlp;
}
