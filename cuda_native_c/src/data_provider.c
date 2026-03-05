/* data_provider.c – DataProvider implementations (plain C99)
 *
 * Image / zip / video backends:
 *   - Images  : stb_image (single-header) + POSIX dirent
 *   - Zip     : system("unzip") + image directory read
 *   - Video   : system("ffmpeg") to extract frames + image directory read
 *   - Synthetic: random float32 RGB
 */
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

#include "data_provider.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#include <math.h>

/* ── Simple bilinear resize ─────────────────────────────────────────────── */
/* Resize uint8 RGB from (sw × sh) to (dw × dh) */
static void bilinear_resize_u8(const unsigned char *src, int sw, int sh,
                                 unsigned char *dst,       int dw, int dh,
                                 int channels)
{
    float x_ratio = (float)(sw - 1) / (float)(dw - 1 > 0 ? dw - 1 : 1);
    float y_ratio = (float)(sh - 1) / (float)(dh - 1 > 0 ? dh - 1 : 1);
    for (int dy = 0; dy < dh; ++dy) {
        float fy = dy * y_ratio;
        int   y0 = (int)fy, y1 = y0 + 1 < sh ? y0 + 1 : y0;
        float ya = fy - y0;
        for (int dx = 0; dx < dw; ++dx) {
            float fx = dx * x_ratio;
            int   x0 = (int)fx, x1 = x0 + 1 < sw ? x0 + 1 : x0;
            float xa = fx - x0;
            for (int c = 0; c < channels; ++c) {
                float p00 = src[(y0*sw + x0)*channels + c];
                float p01 = src[(y0*sw + x1)*channels + c];
                float p10 = src[(y1*sw + x0)*channels + c];
                float p11 = src[(y1*sw + x1)*channels + c];
                float v   = p00*(1-xa)*(1-ya) + p01*xa*(1-ya)
                          + p10*(1-xa)*ya     + p11*xa*ya;
                dst[(dy*dw + dx)*channels + c] = (unsigned char)(v + 0.5f);
            }
        }
    }
}

/* ── String utilities ───────────────────────────────────────────────────── */
static int str_ends_with(const char *s, const char *suffix)
{
    size_t sl = strlen(s), xl = strlen(suffix);
    if (xl > sl) return 0;
    /* case-insensitive compare */
    const char *p = s + sl - xl;
    for (size_t i = 0; i < xl; ++i)
        if ((p[i]|32) != (suffix[i]|32)) return 0;
    return 1;
}

static int is_image_file(const char *name)
{
    return str_ends_with(name, ".jpg")  ||
           str_ends_with(name, ".jpeg") ||
           str_ends_with(name, ".png");
}

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char **)a, *(const char **)b);
}

static int is_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode);
}

/* ── Path list builder ──────────────────────────────────────────────────── */
typedef struct { char **paths; int n; int cap; } PathList;

static void pl_init(PathList *pl) { pl->paths=NULL; pl->n=0; pl->cap=0; }
static void pl_free(PathList *pl) {
    for (int i=0; i<pl->n; ++i) free(pl->paths[i]);
    free(pl->paths); pl_init(pl);
}
static void pl_push(PathList *pl, const char *p) {
    if (pl->n == pl->cap) {
        pl->cap = pl->cap ? pl->cap*2 : 64;
        pl->paths = (char **)realloc(pl->paths, (size_t)pl->cap * sizeof(char*));
    }
    pl->paths[pl->n++] = strdup(p);
}
static void pl_sort(PathList *pl) {
    qsort(pl->paths, (size_t)pl->n, sizeof(char*), cmp_str);
}

/* Recursively collect image files from a directory */
static void pl_scan_dir(PathList *pl, const char *dir)
{
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        if (is_image_file(e->d_name)) {
            pl_push(pl, path);
        } else if (is_dir(path)) {
            pl_scan_dir(pl, path);
        }
    }
    closedir(d);
}

/* ── Simple bilinear resize (for stb_image_resize2 fallback) ───────────── */
/* Uses stbir from stb_image_resize2.h */
static float *decode_and_resize(const char *path, int target_w, int target_h,
                                  int channels)
{
    int w, h, c;
    unsigned char *raw = stbi_load(path, &w, &h, &c, channels);
    if (!raw) {
        fprintf(stderr, "data_provider: cannot load '%s': %s\n",
                path, stbi_failure_reason());
        return NULL;
    }

    float *out = (float *)malloc((size_t)target_w * target_h * channels * sizeof(float));
    if (!out) { stbi_image_free(raw); return NULL; }

    if (w == target_w && h == target_h) {
        for (int i = 0; i < w * h * channels; ++i)
            out[i] = raw[i] / 255.0f;
    } else {
        unsigned char *resized = (unsigned char *)malloc(
            (size_t)target_w * target_h * channels);
        bilinear_resize_u8(raw, w, h, resized, target_w, target_h, channels);
        for (int i = 0; i < target_w * target_h * channels; ++i)
            out[i] = resized[i] / 255.0f;
        free(resized);
    }
    stbi_image_free(raw);
    return out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * IMAGE DIRECTORY PROVIDER
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    DataProvider base;
    PathList     files;
    int          pos;
    float       *frame_buf;   /* [target_w * target_h * 3] */
    char         desc[512];
} ImageDirDP;

static void imgdir_advance(DataProvider *dp)
{
    ImageDirDP *self = (ImageDirDP *)dp;
    if (self->files.n == 0) return;

    free(self->frame_buf);
    self->frame_buf = decode_and_resize(
        self->files.paths[self->pos],
        dp->target_w, dp->target_h, 3);

    if (!self->frame_buf) {
        /* Allocate black frame on decode failure */
        self->frame_buf = (float *)calloc(
            (size_t)dp->target_w * dp->target_h * 3, sizeof(float));
    }
    self->pos = (self->pos + 1) % self->files.n;
}

static const float *imgdir_get_frame(const DataProvider *dp)
{
    return ((const ImageDirDP *)dp)->frame_buf;
}
static int imgdir_total(const DataProvider *dp)
{
    return ((const ImageDirDP *)dp)->files.n;
}
static int imgdir_pos(const DataProvider *dp)
{
    return ((const ImageDirDP *)dp)->pos;
}
static const char *imgdir_desc(const DataProvider *dp)
{
    return ((const ImageDirDP *)dp)->desc;
}
static void imgdir_destroy(DataProvider *dp)
{
    ImageDirDP *self = (ImageDirDP *)dp;
    pl_free(&self->files);
    free(self->frame_buf);
    free(self);
}

DataProvider *dp_image_dir_create(const char *dir, int target_w, int target_h)
{
    ImageDirDP *self = (ImageDirDP *)calloc(1, sizeof(ImageDirDP));
    self->base.advance     = imgdir_advance;
    self->base.get_frame   = imgdir_get_frame;
    self->base.total_frames = imgdir_total;
    self->base.get_pos     = imgdir_pos;
    self->base.describe    = imgdir_desc;
    self->base.destroy     = imgdir_destroy;
    self->base.target_w    = target_w;
    self->base.target_h    = target_h;
    self->pos              = 0;
    self->frame_buf        = NULL;

    pl_init(&self->files);
    pl_scan_dir(&self->files, dir);
    pl_sort(&self->files);

    if (self->files.n == 0) {
        fprintf(stderr, "dp_image_dir: no images found in '%s'\n", dir);
        free(self); return NULL;
    }
    snprintf(self->desc, sizeof(self->desc),
             "ImageDir[%d files, %s]", self->files.n, dir);
    printf("ImageDirDP: %d images from '%s'\n", self->files.n, dir);

    /* Prime the first frame */
    imgdir_advance(&self->base);
    return &self->base;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ZIP PROVIDER  (unzip to temp dir, then use ImageDirDP)
 * ═══════════════════════════════════════════════════════════════════════════ */
DataProvider *dp_zip_create(const char **zip_paths, int n_zips,
                             int target_w, int target_h)
{
    const char *tmpdir = "/tmp/pvm_unzip";
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", tmpdir);
    if (system(cmd) != 0) {
        fprintf(stderr, "dp_zip: cannot create tmpdir\n");
        return NULL;
    }

    for (int i = 0; i < n_zips; ++i) {
        snprintf(cmd, sizeof(cmd),
                 "unzip -o -q '%s' -d '%s' 2>/dev/null || true",
                 zip_paths[i], tmpdir);
        system(cmd);
    }
    return dp_image_dir_create(tmpdir, target_w, target_h);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VIDEO PROVIDER  (ffmpeg extract frames -> tmpdir -> ImageDirDP)
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifdef PVM_VIDEO
DataProvider *dp_video_create(const char **paths, int n_paths,
                               int target_w, int target_h)
{
    const char *tmpdir = "/tmp/pvm_video_frames";
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s' && rm -f '%s'/*.jpg", tmpdir, tmpdir);
    system(cmd);

    int frame_count = 0;
    for (int i = 0; i < n_paths; ++i) {
        snprintf(cmd, sizeof(cmd),
                 "ffmpeg -loglevel error -i '%s' -vf 'scale=%d:%d' "
                 "-start_number %d '%s/%%08d.jpg' 2>/dev/null",
                 paths[i], target_w, target_h, frame_count, tmpdir);
        system(cmd);
        /* Count how many frames were extracted */
        snprintf(cmd, sizeof(cmd), "ls '%s'/*.jpg 2>/dev/null | wc -l", tmpdir);
        FILE *p = popen(cmd, "r");
        if (p) { fscanf(p, "%d", &frame_count); pclose(p); }
    }
    if (frame_count == 0) {
        fprintf(stderr, "dp_video: no frames extracted (ffmpeg not installed?)\n");
        return NULL;
    }
    printf("dp_video: extracted %d frames to '%s'\n", frame_count, tmpdir);
    return dp_image_dir_create(tmpdir, target_w, target_h);
}
#endif /* PVM_VIDEO */

/* ═══════════════════════════════════════════════════════════════════════════
 * SYNTHETIC PROVIDER
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    DataProvider base;
    float       *frame_buf;
    int          pos;
    uint64_t     rng;
} SyntheticDP;

static void synthetic_advance(DataProvider *dp)
{
    SyntheticDP *self = (SyntheticDP *)dp;
    int n = dp->target_w * dp->target_h * 3;
    for (int i = 0; i < n; ++i) {
        self->rng = self->rng * 6364136223846793005ULL + 1442695040888963407ULL;
        self->frame_buf[i] = (float)((self->rng >> 33) & 0xFFFF) / 65535.0f;
    }
    self->pos++;
}
static const float *synthetic_get_frame(const DataProvider *dp)
{
    return ((const SyntheticDP *)dp)->frame_buf;
}
static int synthetic_total(const DataProvider *dp) { (void)dp; return -1; }
static int synthetic_pos(const DataProvider *dp) { return ((const SyntheticDP*)dp)->pos; }
static const char *synthetic_desc(const DataProvider *dp) { (void)dp; return "Synthetic"; }
static void synthetic_destroy(DataProvider *dp) {
    SyntheticDP *self = (SyntheticDP*)dp;
    free(self->frame_buf); free(self);
}

DataProvider *dp_synthetic_create(int target_w, int target_h)
{
    SyntheticDP *self = (SyntheticDP *)calloc(1, sizeof(SyntheticDP));
    self->base.advance      = synthetic_advance;
    self->base.get_frame    = synthetic_get_frame;
    self->base.total_frames = synthetic_total;
    self->base.get_pos      = synthetic_pos;
    self->base.describe     = synthetic_desc;
    self->base.destroy      = synthetic_destroy;
    self->base.target_w     = target_w;
    self->base.target_h     = target_h;
    self->pos               = 0;
    self->rng               = (uint64_t)time(NULL);
    self->frame_buf = (float *)calloc(
        (size_t)target_w * target_h * 3, sizeof(float));
    synthetic_advance(&self->base);  /* prime first frame */
    return &self->base;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * FACTORY / DESTROY
 * ═══════════════════════════════════════════════════════════════════════════ */
DataProvider *dp_create_auto(const char *dataset_name,
                              const char *base_path,
                              const char *file_path,
                              int target_w, int target_h)
{
    char path[4096];

    /* 1. Try dataset_name as subdirectory of base_path */
    if (dataset_name && *dataset_name && base_path && *base_path) {
        snprintf(path, sizeof(path), "%s/%s", base_path, dataset_name);
        if (is_dir(path)) {
            /* Look for zip files first */
            PathList zips; pl_init(&zips);
            DIR *d = opendir(path);
            if (d) {
                struct dirent *e;
                while ((e = readdir(d)) != NULL)
                    if (str_ends_with(e->d_name, ".zip")) {
                        char zp[4096];
                        snprintf(zp, sizeof(zp), "%s/%s", path, e->d_name);
                        pl_push(&zips, zp);
                    }
                closedir(d);
            }
            pl_sort(&zips);
            if (zips.n > 0) {
                DataProvider *dp = dp_zip_create(
                    (const char **)zips.paths, zips.n, target_w, target_h);
                pl_free(&zips);
                return dp;
            }
            pl_free(&zips);
            /* No zips: try as image directory */
            DataProvider *dp = dp_image_dir_create(path, target_w, target_h);
            if (dp) return dp;
            /* Fall through to synthetic */
        }
    }

    /* 2. Try direct file_path */
    if (file_path && *file_path) {
        if (is_dir(file_path)) {
            DataProvider *dp = dp_image_dir_create(file_path, target_w, target_h);
            if (dp) return dp;
            fprintf(stderr, "dp_create_auto: '%s' has no .jpg/.jpeg/.png images, "
                            "falling back to synthetic data.\n", file_path);
            return dp_synthetic_create(target_w, target_h);
        }
        if (str_ends_with(file_path, ".zip")) {
            const char *zp[1] = { file_path };
            return dp_zip_create(zp, 1, target_w, target_h);
        }
#ifdef PVM_VIDEO
        if (str_ends_with(file_path, ".mp4") ||
            str_ends_with(file_path, ".avi") ||
            str_ends_with(file_path, ".mov") ||
            str_ends_with(file_path, ".MOV"))
        {
            const char *vp[1] = { file_path };
            return dp_video_create(vp, 1, target_w, target_h);
        }
#endif
        /* Try as single image directory (basename of file? just try dir) */
        /* Fall through to synthetic */
    }

    /* 3. Synthetic fallback */
    fprintf(stderr, "dp_create_auto: no valid data source; using synthetic data.\n");
    return dp_synthetic_create(target_w, target_h);
}

void dp_destroy(DataProvider *dp)
{
    if (dp && dp->destroy) dp->destroy(dp);
}
