#include "server.hpp"
#include "inference_engine.hpp"
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace arm_ai {

class AIServer::Impl {
public:
    httplib::Server svr;
    InferenceEngine engine;
    int port = 8080;
    bool running = false;
};

AIServer::AIServer() : pimpl(std::make_unique<Impl>()) {}
AIServer::~AIServer() = default;

bool AIServer::start(const std::string& address, int port, int n_ctx, int n_threads) {
    pimpl->port = port;
    // Load model? Actually model path is passed via command line, not here.
    // You'll have to adapt: maybe store model path in a global or pass via start.
    // For now, assume model already loaded in InferenceEngine.
    // Set up routes:
    pimpl->svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        // ... handle request using pimpl->engine
        res.set_header("Access-Control-Allow-Origin", "*");
        // ... 
    });
    pimpl->svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        // ...
    });
    pimpl->svr.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    pimpl->running = true;
    return pimpl->svr.listen(address.c_str(), port);
}

void AIServer::stop() {
    pimpl->svr.stop();
    pimpl->running = false;
}

} // namespace arm_ai
