#include "server.hpp"
#include "inference_engine.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <thread>

using json = nlohmann::json;

namespace arm_ai {

// ---- Constructor & Destructor ----
AIServer::AIServer()
    : server_(std::make_unique<httplib::Server>()),
      engine_(std::make_unique<InferenceEngine>()) {
    setup_routes();
}

AIServer::~AIServer() {
    stop();
}

// ---- Public methods ----
bool AIServer::start(const std::string& model_path, int port, int n_ctx, int n_threads) {
    if (running_) {
        std::cerr << "Server already running" << std::endl;
        return false;
    }

    // Load model
    if (!engine_->load_model(model_path, n_ctx, 0, n_threads)) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return false;
    }

    model_path_ = model_path;
    port_ = port;

    running_ = true;
    std::cout << "Starting server on port " << port << " ..." << std::endl;

    // Run server in a separate thread (blocking call)
    server_thread_ = std::thread([this, port]() {
        server_->listen("0.0.0.0", port);
    });

    // Wait a moment to ensure listen started (optional)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return running_;
}

void AIServer::stop() {
    if (running_.exchange(false)) {
        server_->stop();
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        std::cout << "Server stopped." << std::endl;
    }
}

// ---- Route setup ----
void AIServer::setup_routes() {
    // CORS headers will be set in each handler
    server_->Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handle_chat_completions(req, res);
    });
    server_->Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_->Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res);
    });
}

// ---- Handler: /v1/chat/completions ----
void AIServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");

    try {
        auto body_json = json::parse(req.body);
        ChatRequest request = chat_request_from_json(body_json);

        if (request.stream) {
            // SSE streaming
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            send_sse_stream(res, request);
        } else {
            // Non‑streaming
            std::lock_guard<std::mutex> lock(engine_mutex_);
            ChatResponse response = engine_->generate(request);
            json resp_json = chat_response_to_json(response);
            res.set_content(resp_json.dump(), "application/json");
        }
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

// ---- Handler: /v1/models ----
void AIServer::handle_models(const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    json resp = {
        {"object", "list"},
        {"data", {
            {{"id", model_path_}, {"object", "model"}, {"created", 1234567890}, {"owned_by", "user"}}
        }}
    };
    res.set_content(resp.dump(), "application/json");
}

// ---- Handler: /health ----
void AIServer::handle_health(const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(R"({"status":"ok"})", "application/json");
}

// ---- Helpers for JSON conversion ----
ChatRequest AIServer::chat_request_from_json(const json& j) {
    ChatRequest req;
    req.model = j.value("model", "unknown");
    req.max_tokens = j.value("max_tokens", 50);
    req.temperature = j.value("temperature", 0.7f);
    req.top_p = j.value("top_p", 0.9f);
    req.stream = j.value("stream", false);

    if (j.contains("messages") && j["messages"].is_array()) {
        for (const auto& msg : j["messages"]) {
            std::string role = msg.value("role", "");
            std::string content = msg.value("content", "");
            req.messages.emplace_back(role, content);
        }
    }
    return req;
}

json AIServer::chat_response_to_json(const ChatResponse& response) {
    json j;
    j["id"] = "chatcmpl-" + std::to_string(response.id);
    j["object"] = "chat.completion";
    j["created"] = response.created;
    j["model"] = response.model;
    j["choices"] = {
        {
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", response.content}}},
            {"finish_reason", response.finish_reason}
        }
    };
    return j;
}

// ---- SSE streaming ----
void AIServer::send_sse_stream(httplib::Response& res, const ChatRequest& request) {
    // We need a streaming generator from InferenceEngine.
    // Assume engine_->generate_stream(request) returns a callback or yields tokens.
    // For this implementation, we'll simulate streaming using a simple loop.
    // You should replace with actual engine streaming.

    // For demonstration, we'll split a dummy response into tokens.
    std::string dummy_text = "This is a streamed response from the server.";
    std::istringstream iss(dummy_text);
    std::string word;
    while (iss >> word) {
        json chunk = {{"token", word + " "}, {"finish", false}};
        std::string sse = "data: " + chunk.dump() + "\n\n";
        res.write(sse.c_str(), sse.size());
        // In real streaming you'd wait for new tokens from engine.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // Final chunk
    json final_chunk = {{"finish", true}};
    std::string final_sse = "data: " + final_chunk.dump() + "\n\n";
    res.write(final_sse.c_str(), final_sse.size());
    res.flush();
}

} // namespace arm_ai
