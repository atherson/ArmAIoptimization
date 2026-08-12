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

    // Load model and start server
    bool start(const std::string& model_path, int port = 8080,
               int n_ctx = 4096, int n_threads = -1);

    // Stop the server
    void stop();

    // Check if server is running
    bool is_running() const { return running_; }

private:
    // OpenAI-compatible endpoints
    void setup_routes();

    // POST /v1/chat/completions
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);

    // GET /v1/models
    void handle_models(const httplib::Request& req, httplib::Response& res);

    // GET /health
    void handle_health(const httplib::Request& req, httplib::Response& res);

    // Convert ChatRequest to JSON
    json chat_response_to_json(const ChatResponse& response);

    // Convert ChatRequest from JSON
    ChatRequest chat_request_from_json(const json& j);

    // Generate SSE stream for streaming responses
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
