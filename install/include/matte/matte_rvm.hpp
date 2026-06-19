// RVM matting core (ONNX Runtime). The canonical home for this engine is
// mattemodel — the matte service owns the model; clients (e.g. the OBS camera
// daemon) just talk to it over HTTP. Wraps an ORT session plus the model's
// recurrent state so callers push RGB frames and get an alpha matte back.
// MIT licensed — see LICENSE.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class RvmMatte {
 public:
  // ep: "cuda" | "trt" | "cpu". Throws Ort::Exception on failure.
  RvmMatte(const std::string& model_path, const std::string& ep, float downsample_ratio);

  // src_rgb: tightly packed NCHW float RGB in [0,1], shape 1x3xHxW.
  // Fills pha (H*W floats, alpha in [0,1]) and, if fgr_out != nullptr, the
  // model's despilled foreground (H*W*3 floats, planar RGB). Recurrent state
  // carries across calls — call reset() when the video stream restarts.
  void matte(const float* src_rgb, int W, int H, std::vector<float>& pha_out,
             std::vector<float>* fgr_out = nullptr);

  void reset();
  float downsample_ratio() const { return ratio_; }

 private:
  Ort::Value make_rec_zero();

  Ort::Env env_;
  Ort::SessionOptions so_;
  Ort::Session sess_{nullptr};
  Ort::MemoryInfo mem_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  std::vector<std::string> in_names_, out_names_;
  std::vector<const char*> in_ptrs_, out_ptrs_;
  int pha_idx_ = -1, fgr_idx_ = -1;

  std::array<Ort::Value, 4> rec_{Ort::Value{nullptr}, Ort::Value{nullptr},
                                 Ort::Value{nullptr}, Ort::Value{nullptr}};
  std::array<float, 1> rec_zero_{0.0f};
  std::array<float, 1> ratio_buf_;
  float ratio_;

  // fp16 models (e.g. rvm_resnet50_fp16.onnx) carry half-precision tensors at the
  // I/O boundary; detected from the session so the same code runs either model.
  bool fp16_ = false;        // src + recurrent state are float16
  bool ratio_fp16_ = false;  // downsample_ratio is float16 (RVM keeps it fp32)
  std::vector<uint16_t> src_half_;     // fp32 src -> fp16 staging
  std::array<uint16_t, 1> rec_zero_h_{0};
  std::array<uint16_t, 1> ratio_buf_h_{0};
};
