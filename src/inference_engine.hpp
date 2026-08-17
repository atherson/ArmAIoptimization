#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

namespace arm_ai {

// RAII wrapper for llama_sampler to prevent leaks on exception paths
struct SamplerDeleter {
  void operator()(llama_sampler *s) const noexcept {
    if (s)
      llama_sampler_free(s);
  }
};
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

struct InferenceStats {
  std::chrono::microseconds prompt_processing_time{0};
  std::chrono::microseconds time_to_first_token{0};
  std::chrono::microseconds total_inference_time{0};
  size_t prompt_tokens = 0;
  size_t generated_tokens = 0;
  size_t peak_memory_bytes = 0;

  double tokens_per_second() const noexcept {
    if (total_inference_time.count() == 0)
      return 0.0;
    return static_cast<double>(generated_tokens) * 1'000'000.0 /
           static_cast<double>(total_inference_time.count());
  }

  double prompt_processing_tokens_per_second() const noexcept {
    if (prompt_processing_time.count() == 0)
      return 0.0;
    return static_cast<double>(prompt_tokens) * 1'000'000.0 /
           static_cast<double>(prompt_processing_time.count());
  }
};

struct ChatRequest {
  std::string model;
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

class InferenceEngine {
public:
  InferenceEngine();
  ~InferenceEngine();

  bool load_model(const std::string &model_path, int n_ctx = 4096,
                  int n_threads = -1, int n_gpu_layers = 0);

  bool is_loaded() const noexcept {
    return model_ != nullptr && ctx_ != nullptr;
  }
  std::string get_model_name() const { return model_name_; }

  ChatResponse chat_completion(const ChatRequest &request);

  void chat_completion_stream(const ChatRequest &request,
                              std::function<void(const std::string &, bool,
                                                 bool, const InferenceStats &)>
                                  callback);

  size_t get_peak_memory() const noexcept;

private:
  SamplerPtr make_sampler(float temperature, float top_p) const;

  bool tokenize_prompt(const std::string &prompt,
                       std::vector<llama_token> &out_tokens) const;

  std::string token_to_piece(llama_token id) const;

private:
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{
      nullptr, llama_model_free};
  std::unique_ptr<llama_context, decltype(&llama_free)> ctx_{nullptr,
                                                             llama_free};

  const llama_vocab *vocab_ = nullptr;

  std::string model_name_;
  int n_ctx_ = 4096;
  int n_threads_ = -1;

  std::atomic<size_t> model_weight_bytes_{0};
  std::atomic<size_t> runtime_peak_bytes_{0};

  mutable std::mutex mutex_;
};

} // namespace arm_ai
