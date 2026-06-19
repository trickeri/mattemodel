// mattemodel — system-wide RVM matting daemon.
//
// One RVM model warm on the GPU, exposed over local HTTP so any app (nulpaint
// for Krita subject-select, Kdenlive roto, an OBS client) can POST an image and
// get an alpha matte back. Backend (ncnn-Vulkan or ONNX-CUDA) is chosen at build
// time; this file is backend-agnostic via MatteEngine.
//
// Routes:
//   POST /matte?format=cutout|alpha   multipart "image" -> PNG
//   GET  /status                      JSON: model, backend, loaded, port
//   POST /load /activate              ensure model resident; JSON status
//   POST /unload /park                free the model; JSON status
//
// park == unload here (the RVM model is small; reload from page cache is
// sub-second). The endpoint exists so the Plasma model-manager drives every
// service through one API.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "httplib.h"
#include "matte_engine.hpp"

namespace {

std::string getenv_or(const char* k, const std::string& dflt) {
    const char* v = std::getenv(k);
    return v && *v ? std::string(v) : dflt;
}

std::string status_json(const MatteEngine& eng, const std::string& host, int port) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "{\"service\":\"mattemodel\",\"model\":\"%s\",\"backend\":\"%s\","
        "\"loaded\":%s,\"host\":\"%s\",\"port\":%d}",
        eng.model_path().c_str(), MatteEngine::backend(),
        eng.loaded() ? "true" : "false", host.c_str(), port);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string host  = getenv_or("MM_HOST", "127.0.0.1");
    const int         port  = std::atoi(getenv_or("MM_PORT", "48460").c_str());
    const std::string model = argc > 1 ? argv[1] : getenv_or("MM_MODEL", "");
    if (model.empty()) {
        std::fprintf(stderr, "mattemodel: no model given (arg1 or MM_MODEL)\n");
        return 1;
    }

    if (!MatteEngine::global_init()) {
        std::fprintf(stderr, "mattemodel: backend init failed (%s)\n", MatteEngine::backend());
        return 1;
    }

    MatteEngine engine(model);
    if (!engine.load()) {
        std::fprintf(stderr, "mattemodel: failed to load model '%s' (%s)\n",
                     model.c_str(), MatteEngine::backend());
        MatteEngine::global_cleanup();
        return 1;
    }

    httplib::Server svr;

    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(status_json(engine, host, port), "application/json");
    });

    auto reply = [&](httplib::Response& res) {
        res.set_content(status_json(engine, host, port), "application/json");
    };
    svr.Post("/load",     [&](const httplib::Request&, httplib::Response& res) { engine.load();   reply(res); });
    svr.Post("/activate", [&](const httplib::Request&, httplib::Response& res) { engine.load();   reply(res); });
    svr.Post("/unload",   [&](const httplib::Request&, httplib::Response& res) { engine.unload(); reply(res); });
    svr.Post("/park",     [&](const httplib::Request&, httplib::Response& res) { engine.unload(); reply(res); });

    svr.Post("/matte", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing multipart field 'image'\"}", "application/json");
            return;
        }
        const std::string fmt = req.has_param("format") ? req.get_param_value("format") : "cutout";
        const auto& file = req.get_file_value("image");
        std::vector<uchar> buf(file.content.begin(), file.content.end());
        cv::Mat bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"could not decode image\"}", "application/json");
            return;
        }
        cv::Mat fgr, pha;
        if (!engine.matte(bgr, fgr, pha)) {
            res.status = 500;
            res.set_content("{\"error\":\"matte failed\"}", "application/json");
            return;
        }
        std::vector<uchar> png;
        if (fmt == "alpha") {
            cv::imencode(".png", pha, png);
        } else {
            cv::Mat bgra(bgr.rows, bgr.cols, CV_8UC4);
            for (int y = 0; y < bgr.rows; ++y) {
                const uchar* pf = fgr.ptr<uchar>(y);
                const uchar* pa = pha.ptr<uchar>(y);
                uchar* p = bgra.ptr<uchar>(y);
                for (int x = 0; x < bgr.cols; ++x) {
                    p[0] = pf[0]; p[1] = pf[1]; p[2] = pf[2]; p[3] = pa[0];
                    pf += 3; pa += 1; p += 4;
                }
            }
            cv::imencode(".png", bgra, png);
        }
        res.set_content(reinterpret_cast<const char*>(png.data()), png.size(), "image/png");
    });

    std::fprintf(stderr, "mattemodel: %s on %s:%d  [backend=%s]\n",
                 model.c_str(), host.c_str(), port, MatteEngine::backend());
    bool ok = svr.listen(host.c_str(), port);
    MatteEngine::global_cleanup();
    return ok ? 0 : 1;
}
