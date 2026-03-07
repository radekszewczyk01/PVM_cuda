/* main.c – PVM CUDA C entry point (plain C99)
 * Usage:
 *   pvm_c -S <spec.json> [options]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "pvm_config.h"
#include "pvm_object.h"
#include "data_provider.h"
#include "training_manager.h"
#include "infer_runner.h"

static void usage(const char *prog)
{
    fprintf(stderr,
        "PVM CUDA C – Predictive Vision Model\n"
        "Usage: %s -S <spec.json> [options]\n\n"
        "Training options:\n"
        "  -S, --spec      <file>  Model JSON spec (required for new models)\n"
        "  -L, --load      <file>  Load checkpoint file\n"
        "  -d, --dataset   <name>  Dataset subdirectory name\n"
        "  -p, --path      <dir>   Base data path\n"
        "  -f, --file      <path>  Direct video / image dir / zip path\n"
        "  -b, --batch     <N>     Batch size (default: 1)\n"
        "  -G, --no-graph          Disable CUDA Graph (eager kernel mode)\n"
        "  -s, --save-every <N>    Steps between checkpoints (default: 100000)\n"
        "  -o, --out       <pfx>   Checkpoint file prefix (default: pvm_save)\n"
        "\nInference options (add --infer to switch to forward-only mode):\n"
        "  -I, --infer             Run inference only (no weight updates)\n"
        "  -n, --infer-steps <N>   Number of frames to evaluate (default: 1000)\n"
        "  -D, --out-dir   <dir>   Directory for side-by-side PNG output\n"
        "                          (default: /tmp/pvm_infer, '.' = cwd)\n"
        "  -P, --save-png  <N>     Save a comparison PNG every N frames (default: 10)\n"
        "                          Use 0 to disable PNG output (MAE stats only)\n"
        "  -w, --warmup    <N>     Warm-up frames before measuring MAE (default: 3)\n"
        "\n"
        "  -h, --help              Show this help\n",
        prog);
}

int main(int argc, char **argv)
{
    char spec_file    [1024] = "";
    char load_file    [1024] = "";
    char dataset_name [512]  = "";
    char data_path    [1024] = "";
    char file_path    [1024] = "";
    char save_prefix  [512]  = "pvm_save";
    int  batch_size   = 1;
    int  use_graph    = 1;
    int  save_every   = 100000;
    /* ── Inference mode flags ── */
    int  do_infer      = 0;
    char infer_out_dir [512] = "/tmp/pvm_infer";
    int  infer_steps   = 1000;
    int  infer_save_n  = 10;
    int  infer_warmup  = 3;

    static struct option long_opts[] = {
        {"spec",        required_argument, NULL, 'S'},
        {"load",        required_argument, NULL, 'L'},
        {"dataset",     required_argument, NULL, 'd'},
        {"path",        required_argument, NULL, 'p'},
        {"file",        required_argument, NULL, 'f'},
        {"batch",       required_argument, NULL, 'b'},
        {"no-graph",    no_argument,       NULL, 'G'},
        {"save-every",  required_argument, NULL, 's'},
        {"out",         required_argument, NULL, 'o'},
        /* inference */
        {"infer",       no_argument,       NULL, 'I'},
        {"infer-steps", required_argument, NULL, 'n'},
        {"out-dir",     required_argument, NULL, 'D'},
        {"save-png",    required_argument, NULL, 'P'},
        {"warmup",      required_argument, NULL, 'w'},
        {"help",        no_argument,       NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt, idx = 0;
    while ((opt = getopt_long(argc, argv, "S:L:d:p:f:b:Gs:o:In:D:P:w:h",
                               long_opts, &idx)) != -1) {
        switch (opt) {
            case 'S': strncpy(spec_file,    optarg, sizeof(spec_file)-1);    break;
            case 'L': strncpy(load_file,    optarg, sizeof(load_file)-1);    break;
            case 'd': strncpy(dataset_name, optarg, sizeof(dataset_name)-1); break;
            case 'p': strncpy(data_path,    optarg, sizeof(data_path)-1);    break;
            case 'f': strncpy(file_path,    optarg, sizeof(file_path)-1);    break;
            case 'b': batch_size  = atoi(optarg); break;
            case 'G': use_graph   = 0;            break;
            case 's': save_every  = atoi(optarg); break;
            case 'o': strncpy(save_prefix, optarg, sizeof(save_prefix)-1);   break;
            /* inference */
            case 'I': do_infer    = 1;            break;
            case 'n': infer_steps = atoi(optarg); break;
            case 'D': strncpy(infer_out_dir, optarg, sizeof(infer_out_dir)-1); break;
            case 'P': infer_save_n = atoi(optarg); break;
            case 'w': infer_warmup = atoi(optarg); break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    /* --spec is always required (checkpoint only stores weights, not architecture) */
    if (!spec_file[0]) {
        fprintf(stderr, "Error: --spec is required\n\n");
        usage(argv[0]);
        return 1;
    }
    if (do_infer && !load_file[0]) {
        fprintf(stderr, "Error: --infer requires --load <checkpoint.bin>\n\n");
        usage(argv[0]);
        return 1;
    }

    /* ── Load config ── */
    PVMConfig cfg;
    if (pvm_config_from_file(&cfg, spec_file) != 0) return 1;
    /* Inference mode always uses batch_size=1 (pvm_pop_prediction takes b=0) */
    cfg.batch_size = do_infer ? 1 : ((batch_size > 0) ? batch_size : 1);

    printf("=== PVM CUDA C ===\n");
    pvm_config_print(&cfg);
    printf("CUDA Graph: %s\n", use_graph ? "enabled" : "disabled");
    fflush(stdout);

    /* ── Build PVM object ── */
    PVMObject *pvm = pvm_object_create(&cfg);
    if (!pvm) { fprintf(stderr, "Failed to create PVMObject\n"); return 1; }

    if (load_file[0]) {
        printf("Loading checkpoint '%s'...\n", load_file);
        if (pvm_load(pvm, load_file) != 0) {
            pvm_object_destroy(pvm); return 1;
        }
        printf("Resumed at step %lld\n", (long long)pvm->step);
    }

    /* ── Data provider ── */
    DataProvider *data = dp_create_auto(
        dataset_name[0] ? dataset_name : NULL,
        data_path[0]    ? data_path    : NULL,
        file_path[0]    ? file_path    : NULL,
        pvm_config_input_size(&cfg),
        pvm_config_input_size(&cfg));

    if (!data) {
        fprintf(stderr, "Failed to create data provider\n");
        pvm_object_destroy(pvm); return 1;
    }
    printf("Data: %s\n", data->describe(data));

    if (do_infer) {
        /* ── Inference mode ── */
        InferOptions iopts;
        infer_default_options(&iopts);
        strncpy(iopts.out_dir, infer_out_dir, sizeof(iopts.out_dir) - 1);
        iopts.steps      = infer_steps;
        iopts.save_every = infer_save_n;
        iopts.warmup     = infer_warmup;

        printf("=== Inference mode ===\n");
        fflush(stdout);
        run_infer(pvm, data, &iopts);
    } else {
        /* ── Training mode ── */
        TrainingManagerOptions opts;
        tm_default_options(&opts);
        opts.use_cuda_graph = use_graph;
        opts.save_every     = save_every;
        snprintf(opts.save_prefix, sizeof(opts.save_prefix), "%s", save_prefix);
        snprintf(opts.model_name,  sizeof(opts.model_name),
                 "%s", dataset_name[0] ? dataset_name : "model");

        TrainingManager tm;
        tm_init(&tm, pvm, data, &opts);

        printf("Starting training for %lld steps...\n", (long long)cfg.steps);
        fflush(stdout);
        tm_run(&tm, cfg.steps);

        /* ── Save final checkpoint ── */
        char final_path[512];
        snprintf(final_path, sizeof(final_path), "%s_%s_final.bin",
                 save_prefix,
                 dataset_name[0] ? dataset_name : "model");
        printf("Saving final checkpoint to '%s'...\n", final_path);
        pvm_save(pvm, final_path);
    }

    /* ── Cleanup ── */
    dp_destroy(data);
    pvm_object_destroy(pvm);
    printf("Done.\n");
    return 0;
}
