// PVM CUDA C++ – main entry point
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <memory>
#include <getopt.h>

#include "pvm_config.h"
#include "pvm_object.cuh"
#include "data_provider.h"
#include "training_manager.h"

static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s -S <spec.json> [options]\n"
        "  -S, --spec    <file>   Model JSON spec  (required)\n"
        "  -L, --load    <file>   Load checkpoint\n"
        "  -d, --dataset <name>   Dataset name (subfolder under --path)\n"
        "  -p, --path    <dir>    Base data path\n"
        "  -f, --file    <file>   Direct video/image file path\n"
        "  -b, --batch   <N>      Batch size (default: 1)\n"
        "  -G, --no-graph         Disable CUDA Graph (use eager kernels)\n"
        "  -h, --help             Show this help\n",
        prog);
}

int main(int argc, char** argv)
{
    std::string spec_file;
    std::string load_file;
    std::string dataset_name;
    std::string data_path;
    std::string file_path;
    int         batch_size  = 1;
    bool        use_graph   = true;

    static struct option long_opts[] = {
        {"spec",     required_argument, nullptr, 'S'},
        {"load",     required_argument, nullptr, 'L'},
        {"dataset",  required_argument, nullptr, 'd'},
        {"path",     required_argument, nullptr, 'p'},
        {"file",     required_argument, nullptr, 'f'},
        {"batch",    required_argument, nullptr, 'b'},
        {"no-graph", no_argument,       nullptr, 'G'},
        {"help",     no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt, idx = 0;
    while ((opt = getopt_long(argc, argv, "S:L:d:p:f:b:Gh", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'S': spec_file    = optarg; break;
            case 'L': load_file    = optarg; break;
            case 'd': dataset_name = optarg; break;
            case 'p': data_path    = optarg; break;
            case 'f': file_path    = optarg; break;
            case 'b': batch_size   = std::atoi(optarg); break;
            case 'G': use_graph    = false; break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    if (spec_file.empty()) {
        fprintf(stderr, "Error: --spec is required\n");
        usage(argv[0]);
        return 1;
    }

    // -------------------------------- load config
    PVMConfig config;
    try {
        config = PVMConfig::from_json(spec_file);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to load spec '%s': %s\n", spec_file.c_str(), e.what());
        return 1;
    }
    config.batch_size = batch_size;

    printf("=== PVM CUDA C++ ===\n");
    printf("Spec      : %s\n", spec_file.c_str());
    printf("Batch size: %d\n", batch_size);
    printf("Layers    : %zu\n", config.layer_shapes.size());
    printf("Steps     : %lld\n", config.steps);
    printf("Dataset   : %s\n", dataset_name.empty() ? "(none)" : dataset_name.c_str());
    fflush(stdout);

    // -------------------------------- build PVM
    auto pvm = std::make_shared<PVMObject>(config, config.batch_size);
    if (!load_file.empty()) {
        printf("Loading checkpoint '%s'...\n", load_file.c_str());
        pvm->load(load_file);
        printf("Loaded at step %lld\n", pvm->step());
    }

    // -------------------------------- data provider
    std::shared_ptr<DataProvider> data;
    try {
        data = make_data_provider(dataset_name, data_path, file_path,
                                   config.input_size(), config.input_size());
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to create data provider: %s\n", e.what());
        return 1;
    }

    // -------------------------------- training manager
    TrainingManager::Options opts;
    opts.save_every   = 100000;
    opts.save_prefix  = "pvm_save";
    opts.model_name   = dataset_name.empty() ? "model" : dataset_name;
    opts.use_cuda_graph  = use_graph;

    TrainingManager trainer(pvm, data, opts);

    printf("Starting training for %lld steps...\n", config.steps);
    trainer.run(config.steps);

    printf("\nDone. Saving final checkpoint...\n");
    pvm->save("pvm_final.bin");
    printf("Saved to pvm_final.bin\n");
    return 0;
}
