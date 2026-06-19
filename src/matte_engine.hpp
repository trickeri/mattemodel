// MatteEngine — one RVM matting engine, backend chosen at BUILD time.
//
// Exactly one of matte_engine_ncnn.cpp (BACKEND=vulkan) or matte_engine_ort.cpp
// (BACKEND=cuda) is compiled; both define this same class so server.cpp is
// backend-agnostic. Stills only (zeroed/!reset recurrent state per call).
#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <opencv2/core/core.hpp>

class MatteEngine {
public:
    // model_path meaning is backend-specific:
    //   vulkan → ncnn prefix, loads <path>.ncnn.{param,bin}
    //   cuda   → path to the .onnx file
    explicit MatteEngine(std::string model_path);
    ~MatteEngine();

    // Process-wide backend setup/teardown (ncnn Vulkan instance; no-op for ORT).
    static bool global_init();
    static void global_cleanup();
    static const char* backend();   // "vulkan" | "cuda"

    bool load();          // ensure model resident; idempotent
    void unload();         // free it
    bool loaded() const;
    const std::string& model_path() const { return model_; }

    // One frame: fills fgr (CV_8UC3 BGR) + pha (CV_8UC1 alpha) at input size.
    // Lazily load()s. Thread-safe.
    bool matte(const cv::Mat& bgr, cv::Mat& fgr, cv::Mat& pha);

private:
    struct Impl;
    std::string model_;
    std::unique_ptr<Impl> d_;
    mutable std::mutex mu_;
};
