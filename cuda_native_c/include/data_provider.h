/* data_provider.h – Data provider interface (plain C)
 * Provides video frames as float32 RGB [H, W, 3] buffers.
 *
 * Three backends:
 *   1. Image directory (JPEG/PNG files via stb_image)
 *   2. Synthetic (random frames, useful for benchmarking)
 *   3. Video file via libavformat/libavcodec (compile with -DPVM_VIDEO)
 */
#ifndef DATA_PROVIDER_H
#define DATA_PROVIDER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── DataProvider (polymorphic via function pointers) ──────────────────── */
typedef struct DataProvider DataProvider;

struct DataProvider {
    /* Advance internal cursor to next frame, decode into current buffer.   */
    void (*advance)(DataProvider *dp);

    /* Return pointer to the most recently decoded float32 RGB frame.
     * Buffer size: target_w * target_h * 3 floats, values in [0, 1].      */
    const float *(*get_frame)(const DataProvider *dp);

    /* Total number of frames (−1 if unbounded, e.g., video or synthetic). */
    int  (*total_frames)(const DataProvider *dp);

    /* Current position. */
    int  (*get_pos)(const DataProvider *dp);

    /* Describe the data source (for logging). */
    const char *(*describe)(const DataProvider *dp);

    /* Cleanup. */
    void (*destroy)(DataProvider *dp);

    int target_w, target_h;
};

/* Default advance+get_frame round-trip convenience helper */
static inline const float *dp_next(DataProvider *dp)
{
    dp->advance(dp);
    return dp->get_frame(dp);
}

/* ── Constructors ──────────────────────────────────────────────────────── */

/* Read all JPEG/PNG images from a directory, sorted, loop forever.
 * Returns NULL on error.                                                   */
DataProvider *dp_image_dir_create(const char *dir, int target_w, int target_h);

/* Read images from a NULL-terminated array of zip file paths.
 * Unzips to /tmp/pvm_unzip, then reads as image directory.
 * Returns NULL on error.                                                   */
DataProvider *dp_zip_create(const char **zip_paths, int n_zips,
                             int target_w, int target_h);

#ifdef PVM_VIDEO
/* Read a video file using libavcodec/libavformat (compile with -lpvm_video).
 * Returns NULL on error.                                                   */
DataProvider *dp_video_create(const char **paths, int n_paths,
                               int target_w, int target_h);
#endif

/* Synthetic (random) frames – useful for benchmarking without real data.   */
DataProvider *dp_synthetic_create(int target_w, int target_h);

/* ── Factory ──────────────────────────────────────────────────────────── */
/* Auto-detect type from arguments (mirrors Python ZipCollectionDataProvider usage):
 *   dataset_name + base_path   -> look for zips or images in base_path/dataset_name/
 *   file_path (*.mp4/avi)      -> video (requires PVM_VIDEO)
 *   file_path (directory)      -> image directory
 *   file_path (*.zip)          -> single zip
 *   empty args                 -> synthetic data                            */
DataProvider *dp_create_auto(const char *dataset_name,
                              const char *base_path,
                              const char *file_path,
                              int target_w, int target_h);

/* Destroy any DataProvider created by the above functions. */
void dp_destroy(DataProvider *dp);

#ifdef __cplusplus
}
#endif

#endif /* DATA_PROVIDER_H */
