#pragma once
// Data Provider - reads video/image data for PVM training
// Supports:
//   - ZIP archives containing PKL or image frames (matches Python ZipCollectionDataProvider)
//   - Video files (MP4/AVI) via OpenCV
//   - Image directories

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>

struct Frame {
    cv::Mat image;    // float32 RGB, values in [0,1]
    cv::Mat label;    // optional segmentation label (empty if none)
};

class DataProvider {
public:
    virtual ~DataProvider() = default;

    virtual Frame get_next() = 0;
    virtual void  advance()  = 0;
    virtual int   total_frames() const = 0;
    virtual int   get_pos()     const = 0;
    virtual bool  has_label()   const { return false; }
    virtual std::string describe() const = 0;
};

// ── ZipCollectionDataProvider ─────────────────────────────────────────────────
// Reads a collection of zip files, each containing frames stored as images.
// Matches the Python ZipCollectionDataProvider exactly.
class ZipCollectionDataProvider : public DataProvider {
public:
    ZipCollectionDataProvider(const std::vector<std::string>& zip_paths,
                               int target_w, int target_h);
    ~ZipCollectionDataProvider() override;

    Frame get_next() override;
    void  advance()  override;
    int   total_frames() const override;
    int   get_pos()      const override { return pos_; }
    std::string describe() const override;

private:
    void load_zip(const std::string& path);
    void ensure_loaded(int idx);

    std::vector<std::string> zip_paths_;
    int target_w_, target_h_;

    // All frames kept as compressed jpeg in memory (lazy decode)
    struct FrameEntry {
        std::vector<uint8_t> jpeg_data;
        std::string          label_path;
    };
    std::vector<FrameEntry> entries_;
    int pos_ = 0;
    Frame current_;
};

// ── VideoDataProvider ─────────────────────────────────────────────────────────
// Opens a list of video files via OpenCV.
class VideoDataProvider : public DataProvider {
public:
    VideoDataProvider(const std::vector<std::string>& video_paths,
                      int target_w, int target_h);

    Frame get_next() override;
    void  advance()  override;
    int   total_frames() const override { return -1; /* unbounded */ }
    int   get_pos()      const override { return global_pos_; }
    std::string describe() const override;

private:
    bool open_next_video();

    std::vector<std::string> paths_;
    int  target_w_, target_h_;
    int  file_idx_    = 0;
    int  global_pos_  = 0;
    Frame current_;
    cv::VideoCapture cap_;
};

// ── ImageDirDataProvider ──────────────────────────────────────────────────────
// Reads all images from a directory, loops forever.
class ImageDirDataProvider : public DataProvider {
public:
    ImageDirDataProvider(const std::string& dir, int target_w, int target_h);

    Frame get_next() override;
    void  advance()  override;
    int   total_frames() const override { return (int)files_.size(); }
    int   get_pos()      const override { return pos_; }
    std::string describe() const override;

private:
    std::vector<std::string> files_;
    int target_w_, target_h_;
    int pos_ = 0;
    Frame current_;
};

// ── Factory ───────────────────────────────────────────────────────────────────
std::unique_ptr<DataProvider> make_data_provider(
    const std::string& dataset_name,
    const std::string& base_path,
    const std::string& file_path,
    int target_w, int target_h);
