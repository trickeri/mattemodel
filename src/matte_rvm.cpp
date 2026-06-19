#include "matte_rvm.hpp"

#include <cstring>
#include <stdexcept>

namespace {
// IEEE-754 float32 <-> float16 conversion (no library dependency, so it works
// regardless of the ORT version's Float16_t helpers).
uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t man = x & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;  // too small -> signed zero
    man = (man | 0x800000u) >> (uint32_t)(1 - exp);
    return (uint16_t)(sign | (man >> 13));
  } else if (exp >= 31) {
    return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));  // inf / nan
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; exp--; }
      man &= 0x3ffu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

// Trim ORT's CUDA memory footprint. The defaults are tuned for throughput at any
// cost: EXHAUSTIVE conv-algo search allocates huge scratch workspaces, and the
// power-of-two arena grabs memory in doubling chunks and never releases it --
// together ~5.8 GB for this model. We are camera-bound (matte >>30fps), so we
// trade that unused speed headroom for a much smaller, steady footprint.
void tune_cuda(OrtCUDAProviderOptionsV2* o) {
  const char* keys[] = {
      "arena_extend_strategy",         // request-sized growth, not next-pow2
      "cudnn_conv_algo_search",        // skip the workspace-hungry exhaustive probe
      "cudnn_conv_use_max_workspace",  // don't reserve the max conv workspace
  };
  const char* vals[] = {"kSameAsRequested", "HEURISTIC", "0"};
  Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
      o, keys, vals, sizeof(keys) / sizeof(keys[0])));
}
}  // namespace

RvmMatte::RvmMatte(const std::string& model_path, const std::string& ep,
                   float downsample_ratio)
    : env_(ORT_LOGGING_LEVEL_WARNING, "rvm"), ratio_(downsample_ratio) {
  so_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (ep == "cuda") {
    OrtCUDAProviderOptionsV2* o = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&o));
    tune_cuda(o);
    so_.AppendExecutionProvider_CUDA_V2(*o);
    Ort::GetApi().ReleaseCUDAProviderOptions(o);
  } else if (ep == "trt") {
    OrtTensorRTProviderOptionsV2* t = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&t));
    const char* keys[] = {"trt_engine_cache_enable", "trt_engine_cache_path"};
    const char* vals[] = {"1", "trt_cache"};
    Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(t, keys, vals, 2));
    so_.AppendExecutionProvider_TensorRT_V2(*t);
    Ort::GetApi().ReleaseTensorRTProviderOptions(t);
    OrtCUDAProviderOptionsV2* c = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&c));
    tune_cuda(c);
    so_.AppendExecutionProvider_CUDA_V2(*c);
    Ort::GetApi().ReleaseCUDAProviderOptions(c);
  }

  sess_ = Ort::Session(env_, model_path.c_str(), so_);
  ratio_buf_[0] = ratio_;

  Ort::AllocatorWithDefaultOptions alloc;
  for (size_t i = 0; i < sess_.GetInputCount(); ++i)
    in_names_.emplace_back(sess_.GetInputNameAllocated(i, alloc).get());
  for (size_t i = 0; i < sess_.GetOutputCount(); ++i)
    out_names_.emplace_back(sess_.GetOutputNameAllocated(i, alloc).get());
  for (auto& n : in_names_) in_ptrs_.push_back(n.c_str());
  for (auto& n : out_names_) out_ptrs_.push_back(n.c_str());
  for (size_t i = 0; i < out_names_.size(); ++i) {
    if (out_names_[i] == "pha") pha_idx_ = static_cast<int>(i);
    if (out_names_[i] == "fgr") fgr_idx_ = static_cast<int>(i);
  }
  if (pha_idx_ < 0) throw std::runtime_error("model has no 'pha' output");

  // Detect half-precision I/O so we feed/read the right dtype.
  for (size_t i = 0; i < sess_.GetInputCount(); ++i) {
    auto et = sess_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
    bool half = et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (in_names_[i] == "src") fp16_ = half;
    else if (in_names_[i] == "downsample_ratio") ratio_fp16_ = half;
  }
  reset();
}

Ort::Value RvmMatte::make_rec_zero() {
  static const int64_t shape[4] = {1, 1, 1, 1};
  if (fp16_)
    return Ort::Value::CreateTensor(mem_, rec_zero_h_.data(), sizeof(uint16_t),
                                    shape, 4,
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  return Ort::Value::CreateTensor<float>(mem_, rec_zero_.data(), 1, shape, 4);
}

void RvmMatte::reset() {
  for (auto& r : rec_) r = make_rec_zero();
}

void RvmMatte::matte(const float* src_rgb, int W, int H, std::vector<float>& pha_out,
                     std::vector<float>* fgr_out) {
  const int64_t src_shape[4] = {1, 3, H, W};
  const int64_t ratio_shape[1] = {1};

  std::vector<Ort::Value> ins;
  ins.reserve(in_names_.size());
  const size_t src_count = static_cast<size_t>(3) * H * W;
  for (const auto& name : in_names_) {
    if (name == "src") {
      if (fp16_) {
        src_half_.resize(src_count);
        for (size_t i = 0; i < src_count; ++i) src_half_[i] = f32_to_f16(src_rgb[i]);
        ins.push_back(Ort::Value::CreateTensor(
            mem_, src_half_.data(), src_count * sizeof(uint16_t), src_shape, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
      } else {
        ins.push_back(Ort::Value::CreateTensor<float>(
            mem_, const_cast<float*>(src_rgb), src_count, src_shape, 4));
      }
    } else if (name == "downsample_ratio") {
      if (ratio_fp16_) {
        ratio_buf_h_[0] = f32_to_f16(ratio_);
        ins.push_back(Ort::Value::CreateTensor(
            mem_, ratio_buf_h_.data(), sizeof(uint16_t), ratio_shape, 1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
      } else {
        ins.push_back(Ort::Value::CreateTensor<float>(mem_, ratio_buf_.data(), 1,
                                                      ratio_shape, 1));
      }
    } else {
      int k = (name == "r1i") ? 0 : (name == "r2i") ? 1 : (name == "r3i") ? 2 : 3;
      ins.push_back(std::move(rec_[k]));
    }
  }

  auto outs = sess_.Run(Ort::RunOptions{nullptr}, in_ptrs_.data(), ins.data(),
                        ins.size(), out_ptrs_.data(), out_ptrs_.size());

  // Recurrent feedback.
  for (size_t i = 0; i < out_names_.size(); ++i) {
    const auto& n = out_names_[i];
    if (n == "r1o") rec_[0] = std::move(outs[i]);
    else if (n == "r2o") rec_[1] = std::move(outs[i]);
    else if (n == "r3o") rec_[2] = std::move(outs[i]);
    else if (n == "r4o") rec_[3] = std::move(outs[i]);
  }

  auto read_out = [](Ort::Value& v, std::vector<float>& dst, size_t n) {
    dst.resize(n);
    if (v.GetTensorTypeAndShapeInfo().GetElementType() ==
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      const uint16_t* p = v.GetTensorData<uint16_t>();
      for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(p[i]);
    } else {
      std::memcpy(dst.data(), v.GetTensorData<float>(), n * sizeof(float));
    }
  };

  read_out(outs[pha_idx_], pha_out, static_cast<size_t>(H) * W);
  if (fgr_out && fgr_idx_ >= 0)
    read_out(outs[fgr_idx_], *fgr_out, static_cast<size_t>(3) * H * W);
}
