#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import numpy as np
import pycuda.autoinit  # noqa: F401 - initializes CUDA context
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import pvmcuda_pkg.data as data
import pvmcuda_pkg.datasets as datasets
import pvmcuda_pkg.utils as utils


def format_hms(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:03d}:{m:02d}:{s:06.3f}"


def build_provider(spec_path, dataset_name, data_path):
    with open(spec_path, "r", encoding="utf-8") as f:
        specs = json.load(f)

    input_size = int(specs["layer_shapes"][0]) * int(specs["input_block_size"])

    if dataset_name not in datasets.sets:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    zip_paths = [os.path.join(data_path, name) for name in datasets.sets[dataset_name]]

    t0 = time.perf_counter()
    provider = data.ZipCollectionDataProvider(zip_paths, input_size, input_size)
    init_elapsed = time.perf_counter() - t0

    return provider, input_size, init_elapsed, len(zip_paths)


def run_benchmark(provider, epochs, print_every):
    total_steps = len(provider) * epochs
    if total_steps <= 0:
        raise RuntimeError("Dataset is empty, nothing to benchmark.")

    gpu_buf = None
    start = time.perf_counter()
    cpu_total = 0.0
    gpu_total = 0.0

    for step in range(total_steps):
        t_cpu0 = time.perf_counter()
        frame = provider.get_next()
        tensor = utils.compress_tensor(frame).astype(np.float32, copy=False)
        cpu_total += time.perf_counter() - t_cpu0

        t_gpu0 = time.perf_counter()
        if gpu_buf is None:
            gpu_buf = gpuarray.to_gpu(tensor)
        else:
            gpu_buf.set(tensor)
        cuda.Context.synchronize()
        gpu_total += time.perf_counter() - t_gpu0

        provider.advance()

        if step == 0 or (step + 1) % print_every == 0 or (step + 1) == total_steps:
            elapsed = time.perf_counter() - start
            done = step + 1
            avg_fps = done / max(elapsed, 1e-9)
            progress = 100.0 * done / total_steps
            eta = (total_steps - done) / max(avg_fps, 1e-9)
            cpu_pct = 100.0 * cpu_total / max(elapsed, 1e-9)
            gpu_pct = 100.0 * gpu_total / max(elapsed, 1e-9)
            msg = (
                f"{done:8d}/{total_steps} | {progress:6.2f}% | "
                f"{avg_fps:9.2f} f/s | CPU {cpu_pct:5.1f}% | GPU {gpu_pct:5.1f}% | "
                f"elapsed {format_hms(elapsed)} | ETA {format_hms(eta)}"
            )
            sys.stdout.write("\r\033[2K" + msg)
            sys.stdout.flush()

    cuda.Context.synchronize()
    total_elapsed = time.perf_counter() - start
    final_fps = total_steps / max(total_elapsed, 1e-9)
    sys.stdout.write("\n")

    cpu_avg_ms = 1000.0 * cpu_total / total_steps
    gpu_avg_ms = 1000.0 * gpu_total / total_steps
    other_time = max(0.0, total_elapsed - cpu_total - gpu_total)
    other_pct = 100.0 * other_time / max(total_elapsed, 1e-9)

    print(f"Done. Uploaded {total_steps} frames in {format_hms(total_elapsed)} ({final_fps:.2f} frames/s).")
    print(f"Breakdown per frame: CPU get/prepare {cpu_avg_ms:.4f} ms, GPU upload {gpu_avg_ms:.4f} ms")
    print(
        "Time share: "
        f"CPU {100.0 * cpu_total / max(total_elapsed, 1e-9):.1f}% | "
        f"GPU {100.0 * gpu_total / max(total_elapsed, 1e-9):.1f}% | "
        f"Other {other_pct:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark only image loading + CPU->GPU upload (no training)."
    )
    parser.add_argument("-S", "--spec", default="model_zoo/small.json", help="Model spec JSON file")
    parser.add_argument("-p", "--path", default="./PVM_data/", help="Path to dataset zips")
    parser.add_argument("-d", "--dataset", default="green_ball_training", help="Dataset key from pvmcuda_pkg.datasets")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="How many full passes over the dataset")
    parser.add_argument("--print-every", type=int, default=100, help="Progress refresh interval in frames")
    args = parser.parse_args()

    provider, input_size, init_elapsed, num_files = build_provider(args.spec, args.dataset, args.path)
    print(f"Dataset: {args.dataset}")
    print(f"Zip files: {num_files}")
    print(f"Input size: {input_size}x{input_size}x3")
    print(f"Frames in one epoch: {len(provider)}")
    print(f"Epochs: {args.epochs}")
    print(f"Dataset init/load time: {format_hms(init_elapsed)}")
    print("Starting GPU upload benchmark...")

    run_benchmark(provider, args.epochs, max(1, args.print_every))


if __name__ == "__main__":
    main()
