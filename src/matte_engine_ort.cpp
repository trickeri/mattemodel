// MatteEngine — ONNX Runtime + CUDA backend (BACKEND=cuda). Wraps RvmMatte
// (reused from NulBGRemoval), the engine Troy's live camera daemon trusts.
#include "matte_engine.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include "matte_rvm.hpp"

struct MatteEngine::Impl {
    std::unique_ptr<RvmMatte> rvm;
};

MatteEngine::MatteEngine(std::string model_path)
    : model_(std::move(model_path)), d_(new Impl) {}
MatteEngine::~MatteEngine() { unload(); }

// Device placement is chosen at launch by the model-manager dock via the
// MM_DEVICE env (set to "cpu" when the model is moved to RAM); default GPU/CUDA.
const char* MatteEngine::backend() {
    static const std::string ep = []() {
        const char* d = std::getenv("MM_DEVICE");
        return std::string(d && std::string(d) == "cpu" ? "cpu" : "cuda");
    }();
    return ep.c_str();
}
bool MatteEngine::global_init() { return true; }   // ORT inits per-session
void MatteEngine::global_cleanup() {}

bool MatteEngine::load() {
    std::lock_guard<std::mutex> lk(mu_);
    if (d_->rvm) return true;
    try {
        d_->rvm = std::make_unique<RvmMatte>(model_, backend(), 0.25f);
    } catch (...) {
        return false;
    }
    return true;
}

void MatteEngine::unload() {
    std::lock_guard<std::mutex> lk(mu_);
    d_->rvm.reset();
}

bool MatteEngine::loaded() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<bool>(d_->rvm);
}

bool MatteEngine::matte(const cv::Mat& bgr, cv::Mat& fgr, cv::Mat& pha) {
    if (!loaded() && !load()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (!d_->rvm) return false;

    const int w = bgr.cols, h = bgr.rows;
    const int plane = w * h;

    // BGR uchar HWC -> NCHW RGB float [0,1].
    std::vector<float> src(3 * plane);
    for (int y = 0; y < h; ++y) {
        const uchar* p = bgr.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            src[0 * plane + y * w + x] = p[3 * x + 2] / 255.f;  // R
            src[1 * plane + y * w + x] = p[3 * x + 1] / 255.f;  // G
            src[2 * plane + y * w + x] = p[3 * x + 0] / 255.f;  // B
        }
    }

    d_->rvm->reset();   // stills: don't carry recurrent state between requests
    std::vector<float> pha_f, fgr_f;
    d_->rvm->matte(src.data(), w, h, pha_f, &fgr_f);
    if ((int)pha_f.size() != plane) return false;

    auto clamp8 = [](float v) {
        v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        return (uchar)(v * 255.f + 0.5f);
    };
    pha.create(h, w, CV_8UC1);
    fgr.create(h, w, CV_8UC3);
    const bool have_fgr = (int)fgr_f.size() == 3 * plane;
    for (int y = 0; y < h; ++y) {
        uchar* pp = pha.ptr<uchar>(y);
        uchar* pf = fgr.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            pp[x] = clamp8(pha_f[y * w + x]);
            if (have_fgr) {
                pf[3 * x + 0] = clamp8(fgr_f[2 * plane + y * w + x]);  // B
                pf[3 * x + 1] = clamp8(fgr_f[1 * plane + y * w + x]);  // G
                pf[3 * x + 2] = clamp8(fgr_f[0 * plane + y * w + x]);  // R
            } else {
                pf[3 * x + 0] = bgr.ptr<uchar>(y)[3 * x + 0];
                pf[3 * x + 1] = bgr.ptr<uchar>(y)[3 * x + 1];
                pf[3 * x + 2] = bgr.ptr<uchar>(y)[3 * x + 2];
            }
        }
    }
    return true;
}
