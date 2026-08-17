#include "server.hpp"
#include "inference_engine.hpp"

#include <chrono>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

using json = nlohmann::json;

namespace arm_ai {

AIServer::AIServer()
    : server_(std::make_unique<httplib::Server>()),
      engine_(std::make_unique<InferenceEngine>()) {
  setup_routes();
}

AIServer::~AIServer() { stop(); }

bool AIServer::start(const std::string &model_path, int port, int n_ctx,
                     int n_threads) {
  if (running_.load(std::memory_order_acquire)) {
    std::cerr << "[AIServer] Server is already running\n";
    return false;
  }

  if (!engine_->load_model(model_path, n_ctx, n_threads, 0)) {
    std::cerr << "[AIServer] Failed to load model: " << model_path << '\n';
    return false;
  }

  model_path_ = model_path;
  port_ = port;

  std::cout << "[AIServer] Binding on port " << port << " ...\n";
  running_.store(true, std::memory_order_release);

  server_thread_ = std::thread([this, port]() {
    if (!server_->listen("0.0.0.0", port)) {
      std::cerr << "[AIServer] httplib::listen failed on port " << port << '\n';
      running_.store(false, std::memory_order_release);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  return running_.load(std::memory_order_acquire);
}

void AIServer::stop() {
  if (running_.exchange(false, std::memory_order_acq_rel)) {
    server_->stop();
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    std::cout << "[AIServer] Server stopped.\n";
  }
}

void AIServer::setup_routes() {
  server_->set_payload_max_length(4UL * 1024UL * 1024UL);

  server_->Post("/v1/chat/completions",
                [this](const httplib::Request &req, httplib::Response &res) {
                  handle_chat_completions(req, res);
                });

  server_->Get("/v1/models",
               [this](const httplib::Request &req, httplib::Response &res) {
                 handle_models(req, res);
               });

  server_->Get("/health",
               [this](const httplib::Request &req, httplib::Response &res) {
                 handle_health(req, res);
               });
}

void AIServer::handle_chat_completions(const httplib::Request &req,
                                       httplib::Response &res) {
  try {
    json j = json::parse(req.body);
    ChatRequest request = chat_request_from_json(j);

    if (request.stream) {
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");
      send_sse_stream(res, request);
    } else {
      ChatResponse response = engine_->chat_completion(request);
      res.set_content(chat_response_to_json(response).dump(),
                      "application/json");
    }
  } catch (const json::exception &e) {
    json err = {
        {"error", {{"message", e.what()}, {"type", "invalid_request_error"}}}};
    res.status = 400;
    res.set_content(err.dump(), "application/json");
  } catch (const std::exception &e) {
    json err = {{"error", {{"message", e.what()}, {"type", "server_error"}}}};
    res.status = 500;
    res.set_content(err.dump(), "application/json");
  }
}

void AIServer::handle_models(const httplib::Request &, httplib::Response &res) {
  json j = {{"object", "list"},
            {"data", json::array({{{"id", model_path_},
                                   {"object", "model"},
                                   {"owned_by", "arm-ai"}}})}};
  res.set_content(j.dump(), "application/json");
}

void AIServer::handle_health(const httplib::Request &, httplib::Response &res) {
  json j = {{"status", "ok"}, {"loaded", engine_->is_loaded()}};
  res.set_content(j.dump(), "application/json");
}

json AIServer::chat_response_to_json(const ChatResponse &response) {
  json choices = json::array();
  for (size_t i = 0; i < response.choices.size(); ++i) {
    choices.push_back(
        {{"index", i},
         {"message",
          {{"role", "assistant"}, {"content", response.choices[i].message}}},
         {"finish_reason", "stop"}});
  }

  return {{"id", "chatcmpl-arm"},
          {"object", "chat.completion"},
          {"choices", choices},
          {"usage",
           {{"prompt_tokens", response.stats.prompt_tokens},
            {"completion_tokens", response.stats.generated_tokens},
            {"total_tokens",
             response.stats.prompt_tokens + response.stats.generated_tokens}}}};
}

ChatRequest AIServer::chat_request_from_json(const json &j) {
  ChatRequest req;
  if (j.contains("messages") && j["messages"].is_array()) {
    for (const auto &msg : j["messages"]) {
      if (!msg.is_object())
        continue;
      req.messages.push_back(
          {msg.value("role", "user"), msg.value("content", "")});
    }
  }
  req.max_tokens = j.value("max_tokens", 128);
  req.temperature = j.value("temperature", 0.7f);
  req.top_p = j.value("top_p", 0.95f);
  req.stream = j.value("stream", false);
  return req;
}

void AIServer::send_sse_stream(httplib::Response &res,
                               const ChatRequest &request) {
  res.set_content_provider(
      "text/event-stream",
      [this, request](size_t /*offset*/, httplib::DataSink &sink) -> bool {
        engine_->chat_completion_stream(
            request, [&sink](const std::string &token, bool, bool is_final,
                             const InferenceStats &) {
              if (is_final) {
                const std::string done_msg = "data: [DONE]\n\n";
                sink.write(done_msg.data(), done_msg.size());
                return;
              }

              json chunk = {
                  {"choices", json::array({{{"delta", {{"content", token}}},
                                            {"index", 0},
                                            {"finish_reason", nullptr}}})}};
              const std::string payload = "data: " + chunk.dump() + "\n\n";
              sink.write(payload.data(), payload.size());
            });

        return false;
      });
}

} // namespace arm_ai
