/* training_manager.h – Training loop orchestrator (plain C) */
#ifndef TRAINING_MANAGER_H
#define TRAINING_MANAGER_H

#include "pvm_object.h"
#include "data_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   use_cuda_graph;    /* 1 = capture & replay CUDA graph after warm-up */
    int   save_every;        /* steps between checkpoint saves                 */
    char  save_prefix[256];  /* prefix for checkpoint file names               */
    char  model_name[128];   /* model name tag used in checkpoint filenames    */
    int   warmup_steps;      /* eager steps before CUDA graph capture           */
    int   print_every_ms;    /* console print interval in milliseconds (default 1000) */
} TrainingManagerOptions;

typedef struct {
    PVMObject              *pvm;
    DataProvider           *data;
    TrainingManagerOptions  opts;

    long long counter;
    long long total_steps;
    int       graph_built;

    /* FPS tracking */
    double    fps_avg;
    double    fps_inst;
    long long frames_since_last;

    /* Timing (POSIX clock_gettime) */
    long long t_start_ns;
    long long t_last_ns;
} TrainingManager;

/* Initialise a TrainingManager (does not allocate on heap; pass address of a local). */
void tm_init(TrainingManager *tm,
             PVMObject       *pvm,
             DataProvider    *data,
             const TrainingManagerOptions *opts);

/* Get default options. */
void tm_default_options(TrainingManagerOptions *opts);

/* Run for the given number of steps. Blocks until done. */
void tm_run(TrainingManager *tm, long long steps);

#ifdef __cplusplus
}
#endif

#endif /* TRAINING_MANAGER_H */
