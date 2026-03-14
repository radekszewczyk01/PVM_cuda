/* =========================================================================
 *  pvm_api.cu  --  Public C API (extern "C") for the PVM CUDA backend
 *  -------------------------------------------------------------------
 *  Thin wrappers that cast opaque void* handles to the internal struct
 *  types and delegate to the real implementation.  This is the ONLY
 *  translation unit that the Python ctypes wrapper needs to link against.
 * ========================================================================= */

#include "pvm_api.h"
#include "pvm_model.h"
#include "readout.h"
#include "mlp.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>

/* =========================================================================
 *  Minimal JSON helpers
 *  --------------------
 *  The config JSON uses only string values and one array-of-strings field
 *  ("layer_shapes").  We do NOT need a general-purpose parser -- just
 *  enough to extract quoted keys and values.
 * ========================================================================= */

/* Advance *pos past any whitespace characters. */
static void json_skip_ws(const char *json, int *pos)
{
    while (json[*pos] && isspace((unsigned char)json[*pos]))
        ++(*pos);
}

/* Find the value region for a given "key" inside the JSON string.
 * On success *vstart points to the first character after the colon
 * (with whitespace skipped), and returns 1.  Returns 0 if key not found. */
static int json_find_key(const char *json, const char *key, int *vstart)
{
    /* Build the search needle: "\"key\"" */
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);

    const char *p = strstr(json, needle);
    if (!p) return 0;

    /* Move past the closing quote of the key */
    p += strlen(needle);

    /* Skip whitespace and colon */
    while (*p && (isspace((unsigned char)*p) || *p == ':'))
        ++p;

    *vstart = (int)(p - json);
    return 1;
}

/* Parse a JSON string value at position *vstart (expects opening '"').
 * Copies the content (without quotes) into buf.  Returns length or -1. */
static int json_parse_string(const char *json, int vstart, char *buf, int buf_sz)
{
    int pos = vstart;
    json_skip_ws(json, &pos);
    if (json[pos] != '"') return -1;
    ++pos; /* skip opening quote */

    int len = 0;
    while (json[pos] && json[pos] != '"') {
        if (len < buf_sz - 1)
            buf[len++] = json[pos];
        ++pos;
    }
    buf[len] = '\0';
    return len;
}

/* Parse a bare (unquoted) numeric value at position vstart into buf.
 * Reads digits, sign, decimal point.  Returns length or -1 on empty. */
static int json_parse_bare(const char *json, int vstart, char *buf, int buf_sz)
{
    int pos = vstart;
    json_skip_ws(json, &pos);
    int len = 0;
    while (json[pos] && (isdigit((unsigned char)json[pos]) ||
           json[pos] == '.' || json[pos] == '-' || json[pos] == '+' ||
           json[pos] == 'e' || json[pos] == 'E')) {
        if (len < buf_sz - 1)
            buf[len++] = json[pos];
        ++pos;
    }
    buf[len] = '\0';
    return len > 0 ? len : -1;
}

/* Parse a JSON value (string or bare number) and convert it to int. */
static int json_parse_int(const char *json, const char *key, int default_val)
{
    int vstart;
    if (!json_find_key(json, key, &vstart)) return default_val;
    char buf[64];
    /* Try quoted string first, then bare number */
    if (json_parse_string(json, vstart, buf, sizeof(buf)) >= 0)
        return atoi(buf);
    if (json_parse_bare(json, vstart, buf, sizeof(buf)) >= 0)
        return atoi(buf);
    return default_val;
}

/* Parse a JSON value (string or bare number) and convert it to float. */
static float json_parse_float(const char *json, const char *key, float default_val)
{
    int vstart;
    if (!json_find_key(json, key, &vstart)) return default_val;
    char buf[64];
    /* Try quoted string first, then bare number */
    if (json_parse_string(json, vstart, buf, sizeof(buf)) >= 0)
        return (float)atof(buf);
    if (json_parse_bare(json, vstart, buf, sizeof(buf)) >= 0)
        return (float)atof(buf);
    return default_val;
}

/* Parse "layer_shapes": ["8","4","2","1"] into an int array.
 * Returns the number of elements written, or 0 on failure. */
static int json_parse_int_array(const char *json, const char *key,
                                int *out, int max_elems)
{
    int vstart;
    if (!json_find_key(json, key, &vstart)) return 0;

    int pos = vstart;
    json_skip_ws(json, &pos);
    if (json[pos] != '[') return 0;
    ++pos; /* skip '[' */

    int count = 0;
    while (json[pos] && json[pos] != ']') {
        json_skip_ws(json, &pos);
        if (json[pos] == '"') {
            char buf[64];
            int len = json_parse_string(json, pos, buf, sizeof(buf));
            if (len < 0) break;
            if (count < max_elems)
                out[count++] = atoi(buf);
            /* Advance past the closing quote of this element */
            ++pos; /* skip opening quote */
            pos += len;
            if (json[pos] == '"') ++pos; /* skip closing quote */
        }
        json_skip_ws(json, &pos);
        if (json[pos] == ',') ++pos;
    }
    return count;
}

/* =========================================================================
 *  JSON -> PVMConfig
 * ========================================================================= */

static PVMConfig json_to_config(const char *json)
{
    PVMConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    /* layer_shapes */
    cfg.num_layers = json_parse_int_array(json, "layer_shapes",
                                          cfg.layer_shapes, PVM_MAX_LAYERS);

    /* integer fields */
    cfg.input_block_size              = json_parse_int(json, "input_block_size", 8);
    cfg.hidden_block_size             = json_parse_int(json, "hidden_block_size", 7);
    cfg.input_channels                = json_parse_int(json, "input_channels", 3);
    cfg.lateral_radius                = json_parse_int(json, "lateral_radius", 5);
    cfg.context_exclude_self          = json_parse_int(json, "context_exclude_self", 1);
    cfg.fan_in_square_size            = json_parse_int(json, "fan_in_square_size", 3);
    cfg.fan_in_radius                 = json_parse_int(json, "fan_in_radius", 2);
    cfg.feed_context_in_complex_layer = json_parse_int(json, "feed_context_in_complex_layer", 0);
    cfg.send_context_two_layers_back  = json_parse_int(json, "send_context_two_layers_back", 0);
    cfg.last_layer_context_to_all     = json_parse_int(json, "last_layer_context_to_all", 0);
    cfg.polynomial                    = json_parse_int(json, "polynomial", 0);
    cfg.delay_each_layer_learning     = json_parse_int(json, "delay_each_layer_learning", 5000);
    cfg.delay_final_learning_rate     = json_parse_int(json, "delay_final_learning_rate", 300000);
    cfg.delay_intermediate_learning_rate = json_parse_int(json, "delay_intermediate_learning_rate", 100000);
    cfg.opt_abs_diff                  = json_parse_int(json, "opt_abs_diff", 0);
    cfg.ignore_depth                  = json_parse_int(json, "ignore_depth", 0);

    /* float fields */
    cfg.initial_learning_rate         = json_parse_float(json, "initial_learning_rate", 0.005f);
    cfg.final_learning_rate           = json_parse_float(json, "final_learning_rate", 0.0005f);
    cfg.intermediate_learning_rate    = json_parse_float(json, "intermediate_learning_rate", 0.001f);
    cfg.momentum_val                  = json_parse_float(json, "momentum", 0.5f);

    return cfg;
}

/* =========================================================================
 *  PVM API wrappers
 * ========================================================================= */

extern "C" void* pvm_api_create(const char *config_json, const char *name)
{
    if (!config_json) return nullptr;

    PVMConfig cfg = json_to_config(config_json);
    PVMObject *pvm = pvm_object_create(&cfg, name ? name : "pvm");
    if (!pvm) return nullptr;

    pvm_generate_graph(pvm);
    pvm_generate_memory(pvm);
    pvm_generate_memory_ptrs(pvm);
    pvm_generate_flow_ptrs(pvm);
    pvm_create_gpu_mem(pvm);

    return (void *)pvm;
}

extern "C" void pvm_api_destroy(void *handle)
{
    if (!handle) return;
    pvm_object_destroy((PVMObject *)handle);
}

extern "C" int pvm_api_get_input_shape(void *handle, int *w, int *h, int *ch)
{
    if (!handle) return -1;
    pvm_get_input_shape((PVMObject *)handle, w, h, ch);
    return 0;
}

extern "C" int pvm_api_push_input(void *handle, const float *frame,
                                  int h, int w, int ch)
{
    if (!handle || !frame) return -1;
    pvm_push_input((PVMObject *)handle, frame, h, w, ch);
    return 0;
}

extern "C" int pvm_api_forward(void *handle)
{
    if (!handle) return -1;
    pvm_forward((PVMObject *)handle);
    return 0;
}

extern "C" int pvm_api_backward(void *handle)
{
    if (!handle) return -1;
    pvm_backward((PVMObject *)handle);
    return 0;
}

extern "C" int pvm_api_update_learning_rate(void *handle, float override_rate)
{
    if (!handle) return -1;
    pvm_update_learning_rate((PVMObject *)handle, override_rate);
    return 0;
}

extern "C" int pvm_api_pop_prediction(void *handle, float *out_buf, int delta_step)
{
    if (!handle || !out_buf) return -1;
    pvm_pop_prediction((PVMObject *)handle, out_buf, delta_step);
    return 0;
}

extern "C" int pvm_api_pop_layer(void *handle, unsigned char *out_buf, int layer)
{
    if (!handle || !out_buf) return -1;
    pvm_pop_layer((PVMObject *)handle, out_buf, layer);
    return 0;
}

extern "C" int pvm_api_freeze_learning(void *handle)
{
    if (!handle) return -1;
    pvm_freeze_learning((PVMObject *)handle);
    return 0;
}

extern "C" int pvm_api_unfreeze_learning(void *handle)
{
    if (!handle) return -1;
    pvm_unfreeze_learning((PVMObject *)handle);
    return 0;
}

extern "C" int pvm_api_save(void *handle, const char *filename)
{
    if (!handle || !filename) return -1;
    return pvm_save((PVMObject *)handle, filename);
}

extern "C" void* pvm_api_load(const char *filename)
{
    if (!filename) return nullptr;
    return (void *)pvm_load(filename);
}

extern "C" int pvm_api_get_step(void *handle)
{
    if (!handle) return -1;
    return ((PVMObject *)handle)->step;
}

extern "C" void pvm_api_set_step(void *handle, int step)
{
    if (!handle) return;
    ((PVMObject *)handle)->step = step;
}

extern "C" const char* pvm_api_get_name(void *handle)
{
    if (!handle) return "";
    return ((PVMObject *)handle)->name;
}

extern "C" const char* pvm_api_get_uniq_id(void *handle)
{
    if (!handle) return "";
    return ((PVMObject *)handle)->uniq_id;
}

extern "C" const char* pvm_api_get_device(void *handle)
{
    if (!handle) return "";
    return ((PVMObject *)handle)->device_name;
}

extern "C" float pvm_api_get_learning_rate(void *handle)
{
    if (!handle) return -1.0f;
    return ((PVMObject *)handle)->learning_rate;
}

extern "C" int pvm_api_get_num_layers(void *handle)
{
    if (!handle) return -1;
    return ((PVMObject *)handle)->config.num_layers;
}

extern "C" int pvm_api_get_layer_shape(void *handle, int layer)
{
    if (!handle) return -1;
    PVMObject *pvm = (PVMObject *)handle;
    if (layer < 0 || layer >= pvm->config.num_layers) return -1;
    return pvm->config.layer_shapes[layer];
}

extern "C" int pvm_api_get_total_units(void *handle)
{
    if (!handle) return -1;
    return ((PVMObject *)handle)->total_units;
}

extern "C" const char* pvm_api_get_time_stamp(void *handle)
{
    if (!handle) return "";
    return ((PVMObject *)handle)->time_stamp;
}

extern "C" int pvm_api_get_graph_length(void *handle)
{
    if (!handle) return -1;
    return ((PVMObject *)handle)->total_units;
}

extern "C" int pvm_api_get_layer_ptr(void *handle, int layer)
{
    if (!handle) return -1;
    PVMObject *pvm = (PVMObject *)handle;
    if (layer < 0 || layer > pvm->config.num_layers) return -1;
    return pvm->layer_ptrs[layer];
}

/* =========================================================================
 *  Config getters -- string-keyed lookups into PVMConfig fields
 * ========================================================================= */

extern "C" int pvm_api_get_config_int(void *handle, const char *key)
{
    if (!handle || !key) return -1;
    PVMConfig *c = &((PVMObject *)handle)->config;

    if (strcmp(key, "num_layers") == 0)                      return c->num_layers;
    if (strcmp(key, "input_block_size") == 0)                return c->input_block_size;
    if (strcmp(key, "hidden_block_size") == 0)               return c->hidden_block_size;
    if (strcmp(key, "input_channels") == 0)                  return c->input_channels;
    if (strcmp(key, "lateral_radius") == 0)                  return c->lateral_radius;
    if (strcmp(key, "context_exclude_self") == 0)            return c->context_exclude_self;
    if (strcmp(key, "fan_in_square_size") == 0)              return c->fan_in_square_size;
    if (strcmp(key, "fan_in_radius") == 0)                   return c->fan_in_radius;
    if (strcmp(key, "feed_context_in_complex_layer") == 0)   return c->feed_context_in_complex_layer;
    if (strcmp(key, "send_context_two_layers_back") == 0)    return c->send_context_two_layers_back;
    if (strcmp(key, "last_layer_context_to_all") == 0)       return c->last_layer_context_to_all;
    if (strcmp(key, "polynomial") == 0)                      return c->polynomial;
    if (strcmp(key, "delay_each_layer_learning") == 0)       return c->delay_each_layer_learning;
    if (strcmp(key, "delay_final_learning_rate") == 0)       return c->delay_final_learning_rate;
    if (strcmp(key, "delay_intermediate_learning_rate") == 0)return c->delay_intermediate_learning_rate;
    if (strcmp(key, "opt_abs_diff") == 0)                    return c->opt_abs_diff;
    if (strcmp(key, "ignore_depth") == 0)                    return c->ignore_depth;

    fprintf(stderr, "pvm_api_get_config_int: unknown key \"%s\"\n", key);
    return -1;
}

extern "C" float pvm_api_get_config_float(void *handle, const char *key)
{
    if (!handle || !key) return -1.0f;
    PVMConfig *c = &((PVMObject *)handle)->config;

    if (strcmp(key, "initial_learning_rate") == 0)           return c->initial_learning_rate;
    if (strcmp(key, "final_learning_rate") == 0)             return c->final_learning_rate;
    if (strcmp(key, "intermediate_learning_rate") == 0)      return c->intermediate_learning_rate;
    if (strcmp(key, "momentum") == 0)                        return c->momentum_val;

    fprintf(stderr, "pvm_api_get_config_float: unknown key \"%s\"\n", key);
    return -1.0f;
}

/* =========================================================================
 *  Graph block accessors
 * ========================================================================= */

extern "C" int pvm_api_get_block_n_xs(void *handle, int block_id)
{
    if (!handle) return -1;
    PVMObject *pvm = (PVMObject *)handle;
    if (block_id < 0 || block_id >= pvm->total_units) return -1;
    return pvm->graph[block_id].n_xs;
}

extern "C" int pvm_api_get_block_n_ys(void *handle, int block_id)
{
    if (!handle) return -1;
    PVMObject *pvm = (PVMObject *)handle;
    if (block_id < 0 || block_id >= pvm->total_units) return -1;
    return pvm->graph[block_id].n_ys;
}

extern "C" int pvm_api_get_block_xs(void *handle, int block_id, int *out)
{
    if (!handle || !out) return -1;
    PVMObject *pvm = (PVMObject *)handle;
    if (block_id < 0 || block_id >= pvm->total_units) return -1;
    PVMBlock *blk = &pvm->graph[block_id];
    memcpy(out, blk->xs, blk->n_xs * sizeof(int));
    return blk->n_xs;
}

extern "C" int pvm_api_get_block_ys(void *handle, int block_id, int *out)
{
    if (!handle || !out) return -1;
    PVMObject *pvm = (PVMObject *)handle;
    if (block_id < 0 || block_id >= pvm->total_units) return -1;
    PVMBlock *blk = &pvm->graph[block_id];
    memcpy(out, blk->ys, blk->n_ys * sizeof(int));
    return blk->n_ys;
}

extern "C" int pvm_api_get_block_size(void *handle, int block_id)
{
    if (!handle) return -1;
    PVMObject *pvm = (PVMObject *)handle;
    if (block_id < 0 || block_id >= pvm->total_units) return -1;
    return pvm->graph[block_id].size;
}

extern "C" int pvm_api_get_block_repr_ptr(void *handle, int block_id)
{
    if (!handle) return -1;
    PVMObject *pvm = (PVMObject *)handle;
    if (block_id < 0 || block_id >= pvm->total_units) return -1;
    return pvm->graph[block_id].r_ptr;
}

/* =========================================================================
 *  Readout API wrappers
 * ========================================================================= */

extern "C" void* readout_api_create(void *pvm_handle, int repr_size,
                                    int heatmap_bs)
{
    if (!pvm_handle) return nullptr;
    ReadoutObject *ro = readout_create((PVMObject *)pvm_handle,
                                       repr_size, heatmap_bs);
    return (void *)ro;
}

extern "C" void readout_api_destroy(void *handle)
{
    if (!handle) return;
    readout_destroy((ReadoutObject *)handle);
}

extern "C" int readout_api_copy_data(void *handle)
{
    if (!handle) return -1;
    readout_copy_data((ReadoutObject *)handle);
    return 0;
}

extern "C" int readout_api_forward(void *handle)
{
    if (!handle) return -1;
    readout_forward((ReadoutObject *)handle);
    return 0;
}

extern "C" int readout_api_train(void *handle, const float *label, int h, int w)
{
    if (!handle || !label) return -1;
    readout_train((ReadoutObject *)handle, label, h, w);
    return 0;
}

extern "C" int readout_api_get_heatmap(void *handle, float *out_buf)
{
    if (!handle || !out_buf) return -1;
    readout_get_heatmap((ReadoutObject *)handle, out_buf);
    return 0;
}

extern "C" int readout_api_update_learning_rate(void *handle, float override_rate)
{
    if (!handle) return -1;
    readout_update_learning_rate((ReadoutObject *)handle, override_rate);
    return 0;
}

extern "C" void readout_api_set_pvm(void *handle, void *pvm_handle)
{
    if (!handle || !pvm_handle) return;
    readout_set_pvm((ReadoutObject *)handle, (PVMObject *)pvm_handle);
}

extern "C" int readout_api_save(void *handle, const char *filename)
{
    if (!handle || !filename) return -1;
    return readout_save((ReadoutObject *)handle, filename);
}

extern "C" void* readout_api_load(const char *filename)
{
    if (!filename) return nullptr;
    return (void *)readout_load(filename);
}

extern "C" int readout_api_get_shape(void *handle)
{
    if (!handle) return -1;
    return ((ReadoutObject *)handle)->shape;
}

extern "C" int readout_api_mlp_set_learning_rate(void *handle, float rate)
{
    if (!handle) return -1;
    ReadoutObject *ro = (ReadoutObject *)handle;
    if (!ro->mlp) return -1;
    mlp_set_learning_rate(ro->mlp, rate);
    return 0;
}
