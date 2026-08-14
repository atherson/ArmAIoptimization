#include "server.hpp"
#include "inference_engine.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

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
    if (!engine_->load_model(model_path, n_ctx, n_threads, 0)) {
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

    // Wait a moment to ensure listen started
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
            ChatResponse response = engine_->chat_completion(request);
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
    j["id"] = response.id;  // response.id is already a string
    j["object"] = "chat.completion";
    j["created"] = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch()
                   ).count();
    j["model"] = response.model;

    // Assuming at least one choice
    if (!response.choices.empty()) {
        const auto& choice = response.choices[0];
        j["choices"] = {
            {
                {"index", choice.index},
                {"message", {{"role", "assistant"}, {"content", choice.message}}},
                {"finish_reason", choice.finish_reason}
            }
        };
    } else {
        j["choices"] = json::array();
    }
    return j;
}

// ---- SSE streaming ----
void AIServer::send_sse_stream(httplib::Response& res, const ChatRequest& request) {
    // Use a content provider to stream data incrementally
    // We'll capture a shared pointer to a stream that the callback writes to.
    struct StreamState {
        std::mutex mtx;
        bool done = false;
        std::string buffer;
        httplib::Response* res;
        std::condition_variable cv;
    };

    auto state = std::make_shared<StreamState>();
    state->res = &res;

    // Set the content provider
    res.set_content_provider(
        "text/event-stream",
        [state](size_t offset, httplib::DataSink& sink) {
            std::unique_lock<std::mutex> lock(state->mtx);
            // Wait until there is data or done
            state->cv.wait(lock, [state]() {
                return !state->buffer.empty() || state->done;
            });
            if (state->done && state->buffer.empty()) {
                sink.done();
                return true;
            }
            // Write the current buffer
            sink.write(state->buffer.data(), state->buffer.size());
            state->buffer.clear();
            return true;
        },
        [state](bool success) {
            // Cleanup if needed
        }
    );

    // Now call the streaming generation
    // We'll capture the state to push tokens as they arrive
    engine_->chat_completion_stream(
        request,
        [state](const std::string& token, bool is_first, bool is_final, const InferenceStats& stats) {
            std::lock_guard<std::mutex> lock(state->mtx);
            json chunk;
            if (is_final) {
                chunk = {{"finish", true}};
            } else {
                chunk = {{"token", token}, {"finish", false}};
            }
            std::string sse = "data: " + chunk.dump() + "\n\n";
            state->buffer += sse;
            if (is_final) {
                state->done = true;
            }
            state->cv.notify_one();
        }
    );

    // Wait until streaming is done (the callback above will set done)
    // The content provider will be called asynchronously.
    // We need to block until done, but the callback might already have finished.
    // We'll use a simple loop to wait.
    while (!state->done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

} // namespace arm_ai
