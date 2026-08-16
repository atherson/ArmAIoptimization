#pragma once

// =============================================================================
// server.hpp — Arm AI HTTP Server (OpenAI-compatible REST API)
// C++17 | cpp-httplib | nlohmann/json
//
// Changes from original review:
//   [FIX-10] Removed engine_mutex_: InferenceEngine has its own internal
//            mutex; a second mutex in AIServer provided no protection and
//            confused reviewers.
//   [FIX-12] Payload size limit documented here; enforced in setup_routes().
//   [FIX-13] send_sse_stream content provider now returns false after stream
//            ends to properly signal end-of-stream to cpp-httplib.
// =============================================================================

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "httplib.h"
#include "nlohmann/json.hpp"

#include "inference_engine.hpp"

using json = nlohmann::json;

namespace arm_ai {

class AIServer {
public:
  AIServer();
  ~AIServer();

  // Start the server. Blocks briefly while httplib binds the socket.
  // Returns false if the model cannot be loaded or the port is in use.
  bool start(const std::string &model_path, int port = 8080, int n_ctx = 4096,
             int n_threads = -1);

  void stop();

  bool is_running() const { return running_.load(std::memory_order_acquire); }

private:
  void setup_routes();

  void handle_chat_completions(const httplib::Request &req,
                               httplib::Response &res);
  void handle_models(const httplib::Request &req, httplib::Response &res);
  void handle_health(const httplib::Request &req, httplib::Response &res);

  json chat_response_to_json(const ChatResponse &response);
  ChatRequest chat_request_from_json(const json &j);

  // [FIX-13] Returns false after the stream ends to signal completion.
  void send_sse_stream(httplib::Response &res, const ChatRequest &request);

private:
  std::unique_ptr<httplib::Server> server_;
  std::unique_ptr<InferenceEngine> engine_;

  std::atomic<bool> running_{false};
  std::thread server_thread_;

  std::string model_path_;
  int port_ = 8080;

  // [FIX-10] engine_mutex_ REMOVED — InferenceEngine is self-synchronising.
};

} // namespace arm_ai
