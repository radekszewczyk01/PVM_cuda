/* pvm_config.h – PVM configuration (plain C)
 * Parses a JSON model-zoo spec file into a PVMConfig struct.
 * No external JSON library needed; minimal hand-rolled parser covers this format.
 */
#ifndef PVM_CONFIG_H
#define PVM_CONFIG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of layers supported */
#define PVM_MAX_LAYERS 32

typedef struct {
    /* Architecture */
    int   layer_shapes[PVM_MAX_LAYERS];
    int   num_layers;
    int   hidden_block_size;
    int   input_block_size;
    int   input_channels;
    int   lateral_radius;
    int   fan_in_square_size;
    float fan_in_radius;
    int   context_exclude_self;             /* bool */
    int   send_context_two_layers_back;     /* bool */
    int   last_layer_context_to_all;        /* bool */
    int   feed_context_in_complex_layer;    /* bool */
    int   polynomial;                       /* bool */

    /* Training */
    float     initial_learning_rate;
    float     final_learning_rate;
    float     intermediate_learning_rate;
    float     momentum;
    long long steps;
    int       delay_each_layer_learning;
    long long delay_final_learning_rate;
    long long delay_intermediate_learning_rate;

    /* Batch */
    int batch_size;
} PVMConfig;

/* Returns input image side length in pixels (layer_shapes[0] * input_block_size) */
static inline int pvm_config_input_size(const PVMConfig *c) {
    return c->layer_shapes[0] * c->input_block_size;
}

/* ── JSON loader ──────────────────────────────────────────────────────────── */

/* initialise with sensible defaults */
void pvm_config_init(PVMConfig *c);

/* Load from file.  Returns 0 on success, -1 on error (message written to stderr). */
int pvm_config_from_file(PVMConfig *c, const char *path);

/* Dump to stdout (for debugging) */
void pvm_config_print(const PVMConfig *c);

#ifdef __cplusplus
}
#endif

#endif /* PVM_CONFIG_H */
