#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <mutex>

#include "httplib.h"
#include "nlohmann/json.hpp"

#include "inference_engine.hpp"

using json = nlohmann::json;

namespace arm_ai {

class AIServer {
public:
    AIServer();
    ~AIServer();

    bool start(const std::string& model_path, int port = 8080, int n_ctx = 4096, int n_threads = -1);

    void stop();

    bool is_running() const { return running_; }

private:
    void setup_routes();

    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);

    void handle_models(const httplib::Request& req, httplib::Response& res);

    void handle_health(const httplib::Request& req, httplib::Response& res);

    json chat_response_to_json(const ChatResponse& response);

    ChatRequest chat_request_from_json(const json& j);

    void send_sse_stream(httplib::Response& res, const ChatRequest& request);

private:
    std::unique_ptr<httplib::Server> server_;
    std::unique_ptr<InferenceEngine> engine_;
    
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    
    mutable std::mutex engine_mutex_;
    
    std::string model_path_;
    int port_ = 8080;
};

} 
