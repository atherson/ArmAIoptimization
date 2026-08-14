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

void AIServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    try {
        json j = json::parse(req.body);
        ChatRequest request = chat_request_from_json(j);

        if (request.stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            send_sse_stream(res, request);
        } else {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            ChatResponse response = engine_->chat_completion(request);
            json j_resp = chat_response_to_json(response);
            res.set_content(j_resp.dump(), "application/json");
        }
    } catch (const std::exception& e) {
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
        res.status = 400;
    }
}

void AIServer::handle_models(const httplib::Request&, httplib::Response& res) {
    json j = {
        {"object", "list"},
        {
            "data", json::array({
                {
                    {"id", model_path_},
                    {"object", "model"},
                    {"owned_by", "arm-ai"}
                }
            })
        }
    };
    res.set_content(j.dump(), "application/json");
}

void AIServer::handle_health(const httplib::Request&, httplib::Response& res) {
    json j = {
        {"status", "ok"},
        {"loaded", engine_->is_loaded()}
    };
    res.set_content(j.dump(), "application/json");
}

json AIServer::chat_response_to_json(const ChatResponse& response) {
    json choices = json::array();
    for (size_t i = 0; i < response.choices.size(); ++i) {
        choices.push_back({
            {"index", i},
            {"message", {
                {"role", "assistant"},
                {"content", response.choices[i].message}
            }},
            {"finish_reason", "stop"}
        });
    }

    return {
        {"id", "chatcmpl-arm"},
        {"object", "chat.completion"},
        {"choices", choices},
        {"usage", {
            {"prompt_tokens", response.stats.prompt_tokens},
            {"completion_tokens", response.stats.generated_tokens},
            {"total_tokens", response.stats.prompt_tokens + response.stats.generated_tokens}
        }}
    };
}

ChatRequest AIServer::chat_request_from_json(const json& j) {
    ChatRequest req;
    if (j.contains("messages") && j["messages"].is_array()) {
        for (const auto& msg : j["messages"]) {
            ChatMessage chat_msg;
            chat_msg.role = msg.value("role", "user");
            chat_msg.content = msg.value("content", "");
            req.messages.push_back(chat_msg);
        }
    }
    req.max_tokens = j.value("max_tokens", 128);
    req.temperature = j.value("temperature", 0.7f);
    req.stream = j.value("stream", false);
    return req;
}

void AIServer::send_sse_stream(httplib::Response& res, const ChatRequest& request) {
    res.set_content_provider(
        "text/event-stream",
        [this, request](size_t /*offset*/, httplib::DataSink& sink) {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            
            engine_->chat_completion_stream(
                request,
                [&sink](const std::string& token, bool is_first, bool is_final, const InferenceStats&) {
                    if (is_final) {
                        sink.write("data: [DONE]\n\n", 15);
                        return;
                    }

                    json chunk = {
                        {"choices", json::array({
                            {
                                {"delta", {{"content", token}}},
                                {"index", 0},
                                {"finish_reason", nullptr}
                            }
                        })}
                    };

                    std::string payload = "data: " + chunk.dump() + "\n\n";
                    sink.write(payload.data(), payload.size());
                }
            );

            return true;
        }
    );
}

} // namespace arm_ai
