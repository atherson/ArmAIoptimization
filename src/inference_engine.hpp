#pragma once

// =============================================================================
// inference_engine.hpp — Arm AI Inference Engine
// C++17 | llama.cpp backend | Armv9-A / KleidiAI optimised
//
// Changes from original review:
//   [FIX-2]  Removed five never-defined private helper declarations that would
//            cause link-time errors if ever called.
//   [FIX-3]  Renamed peak_memory_ → model_weight_bytes_ and added a runtime
//            peak tracker (runtime_peak_bytes_) updated after each decode.
//   [FIX-6]  Added SamplerPtr RAII alias used in the .cpp to prevent leaks on
//            exception paths.
//   [NOTE-8] KV-cache / single-sequence limitation documented.
// =============================================================================

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

namespace arm_ai {

// ---------------------------------------------------------------------------
// RAII wrapper for llama_sampler* — prevents leaks on exception paths.
// [FIX-6]
// ---------------------------------------------------------------------------
struct SamplerDeleter {
  void operator()(llama_sampler *s) const noexcept {
    if (s)
      llama_sampler_free(s);
  }
};
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

// ---------------------------------------------------------------------------
// InferenceStats
// ---------------------------------------------------------------------------
struct InferenceStats {
  std::chrono::microseconds prompt_processing_time{0};
  std::chrono::microseconds time_to_first_token{0};
  std::chrono::microseconds total_inference_time{0};
  size_t prompt_tokens = 0;
  size_t generated_tokens = 0;

  // peak_memory_bytes: model-weight bytes at load time + estimated KV cache.
  // [FIX-3] Previously was only model weight; now includes runtime KV
  //         estimate updated after each generation.
  size_t peak_memory_bytes = 0;

  double tokens_per_second() const {
    if (total_inference_time.count() == 0)
      return 0.0;
    return static_cast<double>(generated_tokens) * 1'000'000.0 /
           static_cast<double>(total_inference_time.count());
  }

  double prompt_processing_tokens_per_second() const {
    if (prompt_processing_time.count() == 0)
      return 0.0;
    return static_cast<double>(prompt_tokens) * 1'000'000.0 /
           static_cast<double>(prompt_processing_time.count());
  }
};

// ---------------------------------------------------------------------------
// ChatRequest / ChatChoice / ChatResponse
// ---------------------------------------------------------------------------
struct ChatRequest {
  std::string model;
  // Each pair is {role, content}, e.g. {"user", "Hello"}.
  std::vector<std::pair<std::string, std::string>> messages;
  float temperature = 0.8f;
  float top_p = 0.95f;
  int max_tokens = 1024;
  bool stream = false;
  std::vector<std::string> stop;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
};

struct ChatChoice {
  size_t index = 0;
  std::string message;
  std::string finish_reason;
};

struct ChatResponse {
  std::string id;
  std::string model;
  std::vector<ChatChoice> choices;
  InferenceStats stats;
};

// ---------------------------------------------------------------------------
// InferenceEngine
//
// Thread-safety: chat_completion() and chat_completion_stream() are
// individually thread-safe (guarded by an internal mutex), but they
// serialise all requests. Concurrent callers will queue.
//
// NOTE [NOTE-8]: llama_batch_get_one always uses seq_id=0. Supporting
// true concurrent multi-sequence inference requires llama_batch with
// explicit seq_id fields — a planned future extension.
// ---------------------------------------------------------------------------
class InferenceEngine {
public:
  InferenceEngine();
  ~InferenceEngine();

  // Load a GGUF model. Returns false on failure; object remains in a
  // safe (is_loaded() == false) state so it can be retried or destroyed.
  bool load_model(const std::string &model_path, int n_ctx = 4096,
                  int n_threads = -1, int n_gpu_layers = 0);

  bool is_loaded() const { return model_ != nullptr && ctx_ != nullptr; }
  std::string get_model_name() const { return model_name_; }

  // Blocking inference. Returns a ChatResponse with choices and stats.
  ChatResponse chat_completion(const ChatRequest &request);

  // Streaming inference. Calls callback for each token:
  //   (token_piece, is_first_token, is_final, stats_so_far)
  // callback is invoked from within the engine lock — keep it fast.
  void chat_completion_stream(const ChatRequest &request,
                              std::function<void(const std::string &, bool,
                                                 bool, const InferenceStats &)>
                                  callback);

  // Returns model-weight bytes + estimated KV-cache bytes after the last
  // completed inference call.
  size_t get_peak_memory() const;

private:
  // Build a sampler chain appropriate for the given temperature.
  // Returns an RAII-wrapped sampler. [FIX-4] [FIX-6]
  SamplerPtr make_sampler(float temperature, float top_p) const;

  // Tokenise prompt into the output vector. Returns false on failure.
  bool tokenize_prompt(const std::string &prompt,
                       std::vector<llama_token> &out_tokens) const;

  // Convert a single token id to a UTF-8 string piece.
  // Returns an empty string on failure rather than appending garbage. [FIX-5]
  std::string token_to_piece(llama_token id) const;

private:
  // llama objects — unique_ptr with library-supplied deleters for RAII.
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{
      nullptr, llama_model_free};
  std::unique_ptr<llama_context, decltype(&llama_free)> ctx_{nullptr,
                                                             llama_free};

  // Non-owning pointer; lifetime is tied to model_. Reset in load_model
  // and cleared in destructor order (model_ destroyed after ctx_). [FIX-1]
  const llama_vocab *vocab_ = nullptr;

  std::string model_name_;
  int n_ctx_ = 4096;
  int n_threads_ = -1;

  // [FIX-3] model_weight_bytes_: fixed at load time.
  //         runtime_peak_bytes_: updated after each inference call.
  std::atomic<size_t> model_weight_bytes_{0};
  std::atomic<size_t> runtime_peak_bytes_{0};

  mutable std::mutex mutex_;
};

} // namespace arm_ai
