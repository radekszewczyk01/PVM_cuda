/* pvm_config.c – PVM configuration loader (plain C99) */
#include "pvm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

/* ── defaults ──────────────────────────────────────────────────────────────── */
void pvm_config_init(PVMConfig *c)
{
    memset(c, 0, sizeof(*c));
    c->hidden_block_size                    = 7;
    c->input_block_size                     = 5;
    c->input_channels                       = 3;
    c->lateral_radius                       = 2;
    c->fan_in_square_size                   = 2;
    c->fan_in_radius                        = 2.0f;
    c->context_exclude_self                 = 0;
    c->send_context_two_layers_back         = 0;
    c->last_layer_context_to_all            = 0;
    c->feed_context_in_complex_layer        = 0;
    c->polynomial                           = 1;
    c->initial_learning_rate                = 0.002f;
    c->final_learning_rate                  = 0.00005f;
    c->intermediate_learning_rate           = 0.001f;
    c->momentum                             = 0.1f;
    c->steps                                = 100000000LL;
    c->delay_each_layer_learning            = 1000;
    c->delay_final_learning_rate            = 1000000LL;
    c->delay_intermediate_learning_rate     = 1000000LL;
    c->batch_size                           = 1;
}

/* ── Minimal JSON helpers ───────────────────────────────────────────────────── */
/* We only need to read:
 *   - top-level string values  "key": "value"
 *   - top-level integer values "key": 12345        (bare numbers)
 *   - one array               "layer_shapes": ["8","4",...]
 * The parser is not general-purpose; it is tailored for the PVM config format.
 */

static const char *skip_ws(const char *p)
{
    while (*p && isspace((unsigned char)*p)) ++p;
    return p;
}

/* Read a quoted string.  Returns pointer past the closing quote.
 * Writes at most buf_len-1 bytes into buf and null-terminates. */
static const char *read_string(const char *p, char *buf, int buf_len)
{
    buf[0] = '\0';
    if (*p != '"') return p;
    ++p;  /* skip opening quote */
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p+1)) { ++p; } /* skip escape */
        if (i < buf_len - 1) buf[i++] = *p;
        ++p;
    }
    buf[i] = '\0';
    if (*p == '"') ++p;  /* skip closing quote */
    return p;
}

/* Read an unquoted token (number or identifier) */
static const char *read_token(const char *p, char *buf, int buf_len)
{
    buf[0] = '\0';
    int i = 0;
    while (*p && !isspace((unsigned char)*p) && *p != ',' && *p != '}' && *p != ']') {
        if (i < buf_len - 1) buf[i++] = *p;
        ++p;
    }
    buf[i] = '\0';
    return p;
}

/* Parse the "layer_shapes" array: ["8","4",...] */
static const char *parse_layer_shapes(const char *p, PVMConfig *c)
{
    c->num_layers = 0;
    if (*p != '[') return p;
    ++p;  /* skip '[' */
    while (*p) {
        p = skip_ws(p);
        if (*p == ']') { ++p; break; }
        if (*p == ',') { ++p; continue; }
        char tok[64];
        if (*p == '"') {
            p = read_string(p, tok, sizeof(tok));
        } else {
            p = read_token(p, tok, sizeof(tok));
        }
        if (tok[0] && c->num_layers < PVM_MAX_LAYERS)
            c->layer_shapes[c->num_layers++] = atoi(tok);
    }
    return p;
}

int pvm_config_from_file(PVMConfig *c, const char *path)
{
    pvm_config_init(c);

    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "pvm_config: cannot open '%s': %s\n", path, strerror(errno));
        return -1;
    }

    /* Read entire file into memory */
    fseek(fp, 0, SEEK_END);
    long flen = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc(flen + 1);
    if (!buf) { fclose(fp); return -1; }
    (void)fread(buf, 1, flen, fp);
    buf[flen] = '\0';
    fclose(fp);

    const char *p = buf;

    /* Walk key-value pairs (flat top-level JSON object) */
    while (*p) {
        p = skip_ws(p);
        if (*p == '{' || *p == '}' || *p == ',') { ++p; continue; }
        if (*p != '"') { ++p; continue; }

        /* Read key */
        char key[128];
        p = read_string(p, key, sizeof(key));
        p = skip_ws(p);
        if (*p == ':') ++p;
        p = skip_ws(p);

        /* Read value */
        char val[128];
        if (strcmp(key, "layer_shapes") == 0) {
            p = parse_layer_shapes(p, c);
            continue;
        }

        if (*p == '"') {
            p = read_string(p, val, sizeof(val));
        } else {
            p = read_token(p, val, sizeof(val));
        }
        if (!val[0]) continue;

        /* Map known keys */
        if      (!strcmp(key, "hidden_block_size"))
            c->hidden_block_size = atoi(val);
        else if (!strcmp(key, "input_block_size"))
            c->input_block_size  = atoi(val);
        else if (!strcmp(key, "input_channels"))
            c->input_channels    = atoi(val);
        else if (!strcmp(key, "lateral_radius"))
            c->lateral_radius    = atoi(val);
        else if (!strcmp(key, "fan_in_square_size"))
            c->fan_in_square_size = atoi(val);
        else if (!strcmp(key, "fan_in_radius"))
            c->fan_in_radius     = (float)atof(val);
        else if (!strcmp(key, "context_exclude_self"))
            c->context_exclude_self = atoi(val) || !strcmp(val,"true") || !strcmp(val,"1");
        else if (!strcmp(key, "send_context_two_layers_back"))
            c->send_context_two_layers_back = atoi(val) || !strcmp(val,"true") || !strcmp(val,"1");
        else if (!strcmp(key, "last_layer_context_to_all"))
            c->last_layer_context_to_all = atoi(val) || !strcmp(val,"true") || !strcmp(val,"1");
        else if (!strcmp(key, "feed_context_in_complex_layer"))
            c->feed_context_in_complex_layer = atoi(val) || !strcmp(val,"true") || !strcmp(val,"1");
        else if (!strcmp(key, "polynomial"))
            c->polynomial = atoi(val) || !strcmp(val,"true") || !strcmp(val,"1");
        else if (!strcmp(key, "initial_learning_rate"))
            c->initial_learning_rate = (float)atof(val);
        else if (!strcmp(key, "final_learning_rate"))
            c->final_learning_rate   = (float)atof(val);
        else if (!strcmp(key, "intermediate_learning_rate"))
            c->intermediate_learning_rate = (float)atof(val);
        else if (!strcmp(key, "momentum"))
            c->momentum = (float)atof(val);
        else if (!strcmp(key, "steps"))
            c->steps = atoll(val);
        else if (!strcmp(key, "delay_each_layer_learning"))
            c->delay_each_layer_learning = atoi(val);
        else if (!strcmp(key, "delay_final_learning_rate"))
            c->delay_final_learning_rate = atoll(val);
        else if (!strcmp(key, "delay_intermediate_learning_rate"))
            c->delay_intermediate_learning_rate = atoll(val);
    }

    free(buf);

    if (c->num_layers == 0) {
        fprintf(stderr, "pvm_config: no layer_shapes found in '%s'\n", path);
        return -1;
    }
    return 0;
}

void pvm_config_print(const PVMConfig *c)
{
    printf("=== PVMConfig ===\n");
    printf("  layers          : %d [", c->num_layers);
    for (int i = 0; i < c->num_layers; ++i)
        printf("%d%s", c->layer_shapes[i], i+1<c->num_layers?",":"");
    printf("]\n");
    printf("  hidden_block    : %d\n", c->hidden_block_size);
    printf("  input_block     : %d\n", c->input_block_size);
    printf("  input_channels  : %d\n", c->input_channels);
    printf("  lateral_radius  : %d\n", c->lateral_radius);
    printf("  fan_in_sq_size  : %d\n", c->fan_in_square_size);
    printf("  fan_in_radius   : %.2f\n", c->fan_in_radius);
    printf("  polynomial      : %d\n", c->polynomial);
    printf("  lr_initial      : %.6f\n", c->initial_learning_rate);
    printf("  lr_final        : %.6f\n", c->final_learning_rate);
    printf("  momentum        : %.4f\n", c->momentum);
    printf("  steps           : %lld\n", c->steps);
    printf("  batch_size      : %d\n",   c->batch_size);
}
