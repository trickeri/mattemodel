// MatteEngine — ncnn-Vulkan backend (BACKEND=vulkan). RVM resnet50/mobilenetv3.
#include "matte_engine.hpp"

#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
#include "gpu.h"

struct MatteEngine::Impl {
    bool resnet50 = false;
    std::unique_ptr<ncnn::Net> net;
};

MatteEngine::MatteEngine(std::string model_path)
    : model_(std::move(model_path)), d_(new Impl) {
    d_->resnet50 = model_.find("resnet50") != std::string::npos;
}
MatteEngine::~MatteEngine() { unload(); }

const char* MatteEngine::backend() { return "vulkan"; }

bool MatteEngine::global_init() {
    if (ncnn::get_gpu_count() == 0) return false;
    ncnn::create_gpu_instance();
    return true;
}
void MatteEngine::global_cleanup() { ncnn::destroy_gpu_instance(); }

bool MatteEngine::load() {
    std::lock_guard<std::mutex> lk(mu_);
    if (d_->net) return true;
    auto net = std::make_unique<ncnn::Net>();
    net->opt.use_vulkan_compute = true;
    if (net->load_param((model_ + ".ncnn.param").c_str())) return false;
    if (net->load_model((model_ + ".ncnn.bin").c_str())) return false;
    d_->net = std::move(net);
    return true;
}

void MatteEngine::unload() {
    std::lock_guard<std::mutex> lk(mu_);
    d_->net.reset();
}

bool MatteEngine::loaded() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<bool>(d_->net);
}

bool MatteEngine::matte(const cv::Mat& bgr, cv::Mat& fgr, cv::Mat& pha) {
    if (!loaded() && !load()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (!d_->net) return false;
    const bool resnet50 = d_->resnet50;

    const int w = bgr.cols, h = bgr.rows;
    const int target_size = 512, max_stride = 16;
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

    ncnn::Mat in_pad, in_small_pad;
    int wpad = 0, hpad = 0;
    const bool downsample = std::max(w, h) > target_size;
    if (downsample) {
        int w2 = w, h2 = h;
        float scale = 1.f;
        if (w > h) { scale = (float)target_size / w; w2 = target_size; h2 = (int)(h2 * scale); }
        else       { scale = (float)target_size / h; h2 = target_size; w2 = (int)(w2 * scale); }
        ncnn::Mat in_small = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, w2, h2);
        int w2pad = (w2 + max_stride - 1) / max_stride * max_stride - w2;
        int h2pad = (h2 + max_stride - 1) / max_stride * max_stride - h2;
        ncnn::copy_make_border(in_small, in_small_pad, h2pad / 2, h2pad - h2pad / 2, w2pad / 2, w2pad - w2pad / 2, ncnn::BORDER_CONSTANT, 114.f);
        in_small_pad.substract_mean_normalize(0, norm_vals);
        if (w > h) { wpad = 0; hpad = (int)(in_small_pad.h / scale) - h; }
        else       { hpad = 0; wpad = (int)(in_small_pad.w / scale) - w; }
        ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
        in_pad.substract_mean_normalize(0, norm_vals);
    } else {
        ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
        wpad = (w + max_stride - 1) / max_stride * max_stride - w;
        hpad = (h + max_stride - 1) / max_stride * max_stride - h;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
        in_pad.substract_mean_normalize(0, norm_vals);
        in_small_pad = in_pad;
    }

    ncnn::Mat r1(in_small_pad.w / 2,  in_small_pad.h / 2,  16);
    ncnn::Mat r2(in_small_pad.w / 4,  in_small_pad.h / 4,  resnet50 ? 32 : 20);
    ncnn::Mat r3(in_small_pad.w / 8,  in_small_pad.h / 8,  resnet50 ? 64 : 40);
    ncnn::Mat r4(in_small_pad.w / 16, in_small_pad.h / 16, resnet50 ? 128 : 64);
    r1.fill(0.f); r2.fill(0.f); r3.fill(0.f); r4.fill(0.f);

    ncnn::Extractor ex = d_->net->create_extractor();
    ex.input("in0", in_pad);
    ex.input("in1", in_small_pad);
    ex.input("in2", r1); ex.input("in3", r2); ex.input("in4", r3); ex.input("in5", r4);

    ncnn::Mat out_fgr, out_pha;
    if (downsample) { ex.extract("out2", out_fgr); ex.extract("out3", out_pha); }
    else            { ex.extract("out0", out_fgr); ex.extract("out1", out_pha); }

    const float denorm_vals[3] = {255.f, 255.f, 255.f};
    out_fgr.substract_mean_normalize(0, denorm_vals);
    fgr.create(out_fgr.h, out_fgr.w, CV_8UC3);
    out_fgr.to_pixels(fgr.data, ncnn::Mat::PIXEL_RGB2BGR);
    out_pha.substract_mean_normalize(0, denorm_vals);
    pha.create(out_pha.h, out_pha.w, CV_8UC1);
    out_pha.to_pixels(pha.data, ncnn::Mat::PIXEL_GRAY);

    fgr = fgr(cv::Rect(wpad / 2, hpad / 2, w, h)).clone();
    pha = pha(cv::Rect(wpad / 2, hpad / 2, w, h)).clone();
    return true;
}
