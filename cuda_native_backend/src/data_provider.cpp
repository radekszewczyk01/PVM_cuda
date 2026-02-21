// Data Provider implementations
#include "data_provider.h"
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

namespace fs = std::filesystem;

// ─── ZipCollectionDataProvider ───────────────────────────────────────────────
// Reads zip archives containing frame images.
// Each zip contains files named 000000.jpg, 000001.jpg, ...
// Uses OpenCV VideoCapture with zip URI where supported, or
// extracts to tmpfs for simplicity.

ZipCollectionDataProvider::ZipCollectionDataProvider(
    const std::vector<std::string>& zip_paths, int w, int h)
    : zip_paths_(zip_paths), target_w_(w), target_h_(h)
{
    for (auto& p : zip_paths_) load_zip(p);
    if (entries_.empty())
        throw std::runtime_error("No frames found in zip collection");
    printf("ZipCollectionDataProvider: %zu frames from %zu archives\n",
           entries_.size(), zip_paths_.size());
}

ZipCollectionDataProvider::~ZipCollectionDataProvider() = default;

void ZipCollectionDataProvider::load_zip(const std::string& path)
{
    // Use OpenCV to open each image inside the zip via its path.
    // We enumerate files matching *.jpg / *.png via libzip.
    // For simplicity, we unzip to /tmp/pvm_data/ once and read from there.
    std::string tmpdir = "/tmp/pvm_unzip";
    std::string cmd = "mkdir -p " + tmpdir + " && unzip -o -q '" + path + "' -d " + tmpdir;
    int ret = system(cmd.c_str());
    if (ret != 0) {
        fprintf(stderr, "Warning: unzip of %s failed (code %d)\n", path.c_str(), ret);
        return;
    }

    // Collect all image files in tmpdir
    std::vector<std::string> files;
    for (auto& e : fs::recursive_directory_iterator(tmpdir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());

    for (auto& f : files) {
        FrameEntry fe;
        // Read file bytes for lazy decode
        FILE* fp = fopen(f.c_str(), "rb");
        if (!fp) continue;
        fseek(fp, 0, SEEK_END);
        fe.jpeg_data.resize(ftell(fp));
        fseek(fp, 0, SEEK_SET);
        fread(fe.jpeg_data.data(), 1, fe.jpeg_data.size(), fp);
        fclose(fp);
        entries_.push_back(std::move(fe));
    }
}

Frame ZipCollectionDataProvider::get_next() { return current_; }

void ZipCollectionDataProvider::advance()
{
    auto& e = entries_[pos_];
    std::vector<uint8_t>& data = e.jpeg_data;
    cv::Mat raw = cv::imdecode(cv::Mat(1, (int)data.size(), CV_8UC1, data.data()), cv::IMREAD_COLOR);
    if (raw.empty()) {
        // skip corrupt frame
        pos_ = (pos_ + 1) % (int)entries_.size();
        return;
    }
    cv::Mat resized, rgb, f32;
    cv::resize(raw, resized, {target_w_, target_h_});
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0f / 255.0f);
    current_.image = f32;
    pos_ = (pos_ + 1) % (int)entries_.size();
}

int ZipCollectionDataProvider::total_frames() const { return (int)entries_.size(); }

std::string ZipCollectionDataProvider::describe() const {
    return "ZipCollection[" + std::to_string(entries_.size()) + " frames]";
}

// ─── VideoDataProvider ────────────────────────────────────────────────────────
VideoDataProvider::VideoDataProvider(const std::vector<std::string>& paths, int w, int h)
    : paths_(paths), target_w_(w), target_h_(h)
{
    if (paths_.empty()) throw std::runtime_error("No video paths given");
    open_next_video();
}

bool VideoDataProvider::open_next_video() {
    if (file_idx_ >= (int)paths_.size()) file_idx_ = 0;
    cap_.open(paths_[file_idx_]);
    if (!cap_.isOpened()) {
        fprintf(stderr, "Cannot open video: %s\n", paths_[file_idx_].c_str());
        file_idx_++;
        return false;
    }
    return true;
}

Frame VideoDataProvider::get_next() { return current_; }

void VideoDataProvider::advance()
{
    cv::Mat frame;
    while (!cap_.read(frame)) {
        file_idx_ = (file_idx_ + 1) % (int)paths_.size();
        open_next_video();
    }
    cv::Mat resized, rgb, f32;
    cv::resize(frame, resized, {target_w_, target_h_});
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0f / 255.0f);
    current_.image = f32;
    global_pos_++;
}

std::string VideoDataProvider::describe() const {
    return "Video[" + paths_[file_idx_] + "]";
}

// ─── ImageDirDataProvider ─────────────────────────────────────────────────────
ImageDirDataProvider::ImageDirDataProvider(const std::string& dir, int w, int h)
    : target_w_(w), target_h_(h)
{
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            files_.push_back(e.path().string());
    }
    std::sort(files_.begin(), files_.end());
    if (files_.empty()) throw std::runtime_error("No images in " + dir);
}

Frame ImageDirDataProvider::get_next() { return current_; }

void ImageDirDataProvider::advance()
{
    cv::Mat raw = cv::imread(files_[pos_], cv::IMREAD_COLOR);
    cv::Mat resized, rgb, f32;
    cv::resize(raw, resized, {target_w_, target_h_});
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0f / 255.0f);
    current_.image = f32;
    pos_ = (pos_ + 1) % (int)files_.size();
}

std::string ImageDirDataProvider::describe() const {
    return "ImageDir[" + std::to_string(files_.size()) + " files]";
}

// ─── Factory ─────────────────────────────────────────────────────────────────
std::unique_ptr<DataProvider> make_data_provider(
    const std::string& dataset_name,
    const std::string& base_path,
    const std::string& file_path,
    int target_w, int target_h)
{
    // Map dataset names to zip file lists (same as Python datasets.py)
    // green_ball_training -> *.zip in base_path/green_ball_training/
    auto collect_zips = [](const std::string& dir) {
        std::vector<std::string> v;
        if (!fs::exists(dir)) return v;
        for (auto& e : fs::directory_iterator(dir)) {
            if (e.path().extension() == ".zip")
                v.push_back(e.path().string());
        }
        std::sort(v.begin(), v.end());
        return v;
    };

    if (!dataset_name.empty()) {
        // Try as directory of zips
        std::string dir = base_path + dataset_name;
        auto zips = collect_zips(dir);
        if (!zips.empty())
            return std::make_unique<ZipCollectionDataProvider>(zips, target_w, target_h);

        // Try base_path directly
        zips = collect_zips(base_path);
        if (!zips.empty())
            return std::make_unique<ZipCollectionDataProvider>(zips, target_w, target_h);
    }

    if (!file_path.empty()) {
        std::string ext = fs::path(file_path).extension();
        if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".MOV")
            return std::make_unique<VideoDataProvider>(
                std::vector<std::string>{file_path}, target_w, target_h);
        if (ext == ".zip")
            return std::make_unique<ZipCollectionDataProvider>(
                std::vector<std::string>{file_path}, target_w, target_h);
        if (fs::is_directory(file_path))
            return std::make_unique<ImageDirDataProvider>(file_path, target_w, target_h);
    }

    throw std::runtime_error("Cannot determine data source. Provide -d or -f argument.");
}
