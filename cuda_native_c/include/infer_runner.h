/* infer_runner.h – Forward-only inference loop for PVM CUDA C
 *
 * Loads a pre-trained checkpoint, runs the forward pass on a data stream,
 * computes Mean Absolute Error (MAE) of the prediction vs actual frame,
 * and saves side-by-side PNG frames (input | prediction) to disk.
 */
#ifndef INFER_RUNNER_H
#define INFER_RUNNER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Options for the inference runner */
typedef struct {
    char out_dir   [512]; /* Directory to write PNGs into (created if missing) */
    int  steps;           /* Number of frames to run (0 = run until data ends)  */
    int  save_every;      /* Save a PNG every N steps (0 = no saving)           */
    int  warmup;          /* Warm-up steps (run forward but skip MAE/save)      */
} InferOptions;

/* Set defaults: /tmp/pvm_infer, 1000 frames, save every 10, 3 warm-up */
void infer_default_options(InferOptions *opts);

/* Run forward-only inference.
 *   pvm  – PVMObject already loaded with checkpoint weights
 *   data – DataProvider pointing at the test sequence
 *   opts – Options (may be NULL for defaults)
 *
 * Returns average MAE over processed frames (excluding warm-up).
 */
double run_infer(void *pvm, void *data, const InferOptions *opts);

#ifdef __cplusplus
}
#endif

#endif /* INFER_RUNNER_H */
