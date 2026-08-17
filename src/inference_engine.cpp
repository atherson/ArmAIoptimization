#include "inference_engine.hpp"

#include <chrono>
#include <iostream>
#include <thread>

namespace arm_ai {

InferenceEngine::InferenceEngine() = default;
InferenceEngine::~InferenceEngine() = default;

bool InferenceEngine::load_model(const std::string &model_path, int n_ctx,
                                 int n_threads, int n_gpu_layers) {
  ctx_.reset();
  model_.reset();
  vocab_ = nullptr;
  model_weight_bytes_.store(0, std::memory_order_relaxed);
  runtime_peak_bytes_.store(0, std::memory_order_relaxed);

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = n_gpu_layers;

  llama_model *raw_model =
      llama_load_model_from_file(model_path.c_str(), model_params);
  if (!raw_model) {
    std::cerr << "[InferenceEngine] Failed to load model from: " << model_path
              << '\n';
    return false;
  }
  model_.reset(raw_model);
  vocab_ = llama_model_get_vocab(model_.get());

  const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
  const int threads = (n_threads > 0) ? n_threads : hw_threads;

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = n_ctx;
  ctx_params.n_batch = 512;
  ctx_params.n_threads = threads;
  ctx_params.n_threads_batch = threads;

  llama_context *raw_ctx = llama_init_from_model(model_.get(), ctx_params);
  if (!raw_ctx) {
    std::cerr << "[InferenceEngine] Failed to create llama context\n";
    vocab_ = nullptr;
    model_.reset();
    return false;
  }
  ctx_.reset(raw_ctx);

  model_name_ = model_path;
  n_ctx_ = n_ctx;
  n_threads_ = threads;

  model_weight_bytes_.store(llama_model_size(model_.get()),
                            std::memory_order_relaxed);
  return true;
}

size_t InferenceEngine::get_peak_memory() const noexcept {
  if (!model_)
    return 0;
  return model_weight_bytes_.load(std::memory_order_relaxed) +
         runtime_peak_bytes_.load(std::memory_order_relaxed);
}

SamplerPtr InferenceEngine::make_sampler(float temperature, float top_p) const {
  SamplerPtr chain(
      llama_sampler_chain_init(llama_sampler_chain_default_params()));

  if (temperature <= 0.0f) {
    llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
  } else {
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(chain.get(),
                            llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
  }
  return chain;
}

bool InferenceEngine::tokenize_prompt(
    const std::string &prompt, std::vector<llama_token> &out_tokens) const {
  out_tokens.resize(prompt.size() + 64);
  int n = llama_tokenize(model_.get(), prompt.c_str(),
                         static_cast<int>(prompt.size()), out_tokens.data(),
                         static_cast<int>(out_tokens.size()), true, true);
  if (n < 0) {
    out_tokens.resize(static_cast<size_t>(-n));
    n = llama_tokenize(model_.get(), prompt.c_str(),
                       static_cast<int>(prompt.size()), out_tokens.data(),
                       static_cast<int>(out_tokens.size()), true, true);
    if (n < 0)
      return false;
  }
  out_tokens.resize(static_cast<size_t>(n));
  return true;
}

std::string InferenceEngine::token_to_piece(llama_token id) const {
  char buf[256];
  int n = llama_token_to_piece(model_.get(), id, buf, sizeof(buf), 0, true);
  if (n > 0) {
    return std::string(buf, static_cast<size_t>(n));
  }
  if (n < 0) {
    std::vector<char> large_buf(static_cast<size_t>(-n));
    int n2 = llama_token_to_piece(model_.get(), id, large_buf.data(),
                                  large_buf.size(), 0, true);
    if (n2 > 0)
      return std::string(large_buf.data(), static_cast<size_t>(n2));
  }
  return {};
}

ChatResponse InferenceEngine::chat_completion(const ChatRequest &request) {
  std::lock_guard<std::mutex> lock(mutex_);
  ChatResponse response;

  if (!is_loaded())
    return response;

  llama_kv_cache_clear(ctx_.get());

  std::string prompt;
  prompt.reserve(512);
  for (const auto &[role, content] : request.messages) {
    prompt += role + ": " + content + '\n';
  }
  prompt += "assistant: ";

  std::vector<llama_token> tokens;
  if (!tokenize_prompt(prompt, tokens))
    return response;

  InferenceStats stats;
  stats.prompt_tokens = tokens.size();

  const auto t_prompt_start = std::chrono::steady_clock::now();
  llama_batch batch =
      llama_batch_get_one(tokens.data(), static_cast<int>(tokens.size()));
  if (llama_decode(ctx_.get(), batch) != 0)
    return response;

  const auto t_prompt_end = std::chrono::steady_clock::now();
  stats.prompt_processing_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_prompt_end -
                                                            t_prompt_start);

  SamplerPtr smpl = make_sampler(request.temperature, request.top_p);
  std::string generated_text;
  generated_text.reserve(static_cast<size_t>(request.max_tokens) * 4);

  const auto t_gen_start = std::chrono::steady_clock::now();
  bool first_token = true;

  for (int i = 0; i < request.max_tokens; ++i) {
    llama_token id = llama_sampler_sample(smpl.get(), ctx_.get(), -1);
    llama_sampler_accept(smpl.get(), id);

    if (llama_token_is_eog(model_.get(), id))
      break;

    if (first_token) {
      stats.time_to_first_token =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - t_prompt_end);
      first_token = false;
    }

    generated_text += token_to_piece(id);
    stats.generated_tokens++;

    batch = llama_batch_get_one(&id, 1);
    if (llama_decode(ctx_.get(), batch) != 0)
      break;
  }

  stats.total_inference_time =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - t_gen_start);

  runtime_peak_bytes_.store(llama_state_get_size(ctx_.get()),
                            std::memory_order_relaxed);
  stats.peak_memory_bytes = get_peak_memory();

  ChatChoice choice;
  choice.index = 0;
  choice.message = std::move(generated_text);
  choice.finish_reason = "stop";
  response.choices.push_back(std::move(choice));
  response.stats = stats;

  return response;
}

void InferenceEngine::chat_completion_stream(
    const ChatRequest &request,
    std::function<void(const std::string &, bool, bool, const InferenceStats &)>
        callback) {

  std::lock_guard<std::mutex> lock(mutex_);
  InferenceStats stats;

  if (!is_loaded()) {
    callback("", false, true, stats);
    return;
  }

  llama_kv_cache_clear(ctx_.get());

  std::string prompt;
  for (const auto &[role, content] : request.messages) {
    prompt += role + ": " + content + '\n';
  }
  prompt += "assistant: ";

  std::vector<llama_token> tokens;
  if (!tokenize_prompt(prompt, tokens)) {
    callback("", false, true, stats);
    return;
  }

  stats.prompt_tokens = tokens.size();
  const auto t_prompt_start = std::chrono::steady_clock::now();
  llama_batch batch =
      llama_batch_get_one(tokens.data(), static_cast<int>(tokens.size()));
  if (llama_decode(ctx_.get(), batch) != 0) {
    callback("", false, true, stats);
    return;
  }

  const auto t_prompt_end = std::chrono::steady_clock::now();
  stats.prompt_processing_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_prompt_end -
                                                            t_prompt_start);

  SamplerPtr smpl = make_sampler(request.temperature, request.top_p);
  const auto t_gen_start = std::chrono::steady_clock::now();
  bool is_first = true;

  for (int i = 0; i < request.max_tokens; ++i) {
    llama_token id = llama_sampler_sample(smpl.get(), ctx_.get(), -1);
    llama_sampler_accept(smpl.get(), id);

    if (llama_token_is_eog(model_.get(), id))
      break;

    if (is_first) {
      stats.time_to_first_token =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - t_prompt_end);
    }

    const std::string piece = token_to_piece(id);
    stats.generated_tokens++;
    callback(piece, is_first, false, stats);
    is_first = false;

    batch = llama_batch_get_one(&id, 1);
    if (llama_decode(ctx_.get(), batch) != 0)
      break;
  }

  stats.total_inference_time =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - t_gen_start);
  runtime_peak_bytes_.store(llama_state_get_size(ctx_.get()),
                            std::memory_order_relaxed);
  stats.peak_memory_bytes = get_peak_memory();
  callback("", false, true, stats);
}

} // namespace arm_ai
