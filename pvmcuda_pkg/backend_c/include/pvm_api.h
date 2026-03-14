#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* -- PVM Object -- */
void* pvm_api_create(const char *config_json, const char *name);
void  pvm_api_destroy(void *handle);
int   pvm_api_get_input_shape(void *handle, int *w, int *h, int *ch);
int   pvm_api_push_input(void *handle, const float *frame, int h, int w, int ch);
int   pvm_api_forward(void *handle);
int   pvm_api_backward(void *handle);
int   pvm_api_update_learning_rate(void *handle, float override_rate);
int   pvm_api_pop_prediction(void *handle, float *out_buf, int delta_step);
int   pvm_api_pop_layer(void *handle, unsigned char *out_buf, int layer);
int   pvm_api_freeze_learning(void *handle);
int   pvm_api_unfreeze_learning(void *handle);
int   pvm_api_save(void *handle, const char *filename);
void* pvm_api_load(const char *filename);
int   pvm_api_get_step(void *handle);
void  pvm_api_set_step(void *handle, int step);
const char* pvm_api_get_name(void *handle);
const char* pvm_api_get_uniq_id(void *handle);
const char* pvm_api_get_device(void *handle);
float pvm_api_get_learning_rate(void *handle);
int   pvm_api_get_num_layers(void *handle);
int   pvm_api_get_layer_shape(void *handle, int layer);
int   pvm_api_get_total_units(void *handle);
const char* pvm_api_get_time_stamp(void *handle);
int   pvm_api_get_config_int(void *handle, const char *key);
float pvm_api_get_config_float(void *handle, const char *key);
int   pvm_api_get_graph_length(void *handle);

/* Layer pointer for pop_layer */
int   pvm_api_get_layer_ptr(void *handle, int layer);

/* Graph block info for readout */
int   pvm_api_get_block_n_xs(void *handle, int block_id);
int   pvm_api_get_block_n_ys(void *handle, int block_id);
int   pvm_api_get_block_xs(void *handle, int block_id, int *out);
int   pvm_api_get_block_ys(void *handle, int block_id, int *out);
int   pvm_api_get_block_size(void *handle, int block_id);
int   pvm_api_get_block_repr_ptr(void *handle, int block_id);

/* -- Readout Object -- */
void* readout_api_create(void *pvm_handle, int repr_size, int heatmap_bs);
void  readout_api_destroy(void *handle);
int   readout_api_copy_data(void *handle);
int   readout_api_forward(void *handle);
int   readout_api_train(void *handle, const float *label, int h, int w);
int   readout_api_get_heatmap(void *handle, float *out_buf);
int   readout_api_update_learning_rate(void *handle, float override_rate);
void  readout_api_set_pvm(void *handle, void *pvm_handle);
int   readout_api_save(void *handle, const char *filename);
void* readout_api_load(const char *filename);
int   readout_api_get_shape(void *handle);
int   readout_api_mlp_set_learning_rate(void *handle, float rate);

#ifdef __cplusplus
}
#endif
