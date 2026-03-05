from pvmcuda_pkg.gpu_routines import initialize_gpu, transfer_data_to_gpu, perform_gpu_computation, transfer_data_to_cpu, cleanup_gpu

# Initialize GPU
initialize_gpu()

# Transfer data to GPU
gpu_data = transfer_data_to_gpu(data)

# Perform computation
result = perform_gpu_computation(gpu_data)

# Transfer result back to CPU
final_result = transfer_data_to_cpu(result)

# Cleanup GPU resources
cleanup_gpu()