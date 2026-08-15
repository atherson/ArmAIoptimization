#include "inference_engine.hpp"
#include <chrono>
#include <iostream>
#include <thread>

namespace arm_ai {

InferenceEngine::InferenceEngine() {}

InferenceEngine::~InferenceEngine() {}

bool InferenceEngine::load_model(const std::string &model_path, int n_ctx,
                                 int n_threads, int n_gpu_layers) {
  model_name_ = model_path;
  n_ctx_ = n_ctx;
  n_threads_ = n_threads;

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = n_gpu_layers;

  struct llama_model *model_ptr =
      llama_load_model_from_file(model_path.c_str(), model_params);
  if (!model_ptr) {
    std::cerr << "Failed to load model from " << model_path << std::endl;
    return false;
  }
  model_.reset(model_ptr);

  vocab_ = llama_model_get_vocab(model_.get());

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = n_ctx;
  ctx_params.n_batch = 512;
  ctx_params.n_threads =
      n_threads > 0 ? n_threads
                    : static_cast<int>(std::thread::hardware_concurrency());
  ctx_params.n_threads_batch = ctx_params.n_threads;

  struct llama_context *ctx_ptr =
      llama_init_from_model(model_.get(), ctx_params);
  if (!ctx_ptr) {
    std::cerr << "Failed to create context" << std::endl;
    return false;
  }
  ctx_.reset(ctx_ptr);

  peak_memory_ = llama_model_size(model_.get());
  return true;
}

size_t InferenceEngine::get_peak_memory() const {
  if (!model_)
    return 0;
  return llama_model_size(model_.get());
}

ChatResponse InferenceEngine::chat_completion(const ChatRequest &request) {
  // Thread safety: Protect the shared llama_context from concurrent accesses
  std::lock_guard<std::mutex> lock(mutex_);

  ChatResponse response;

  if (!is_loaded()) {
    std::cerr << "Model not loaded" << std::endl;
    return response;
  }

  llama_kv_cache_clear(ctx_.get());

  std::string prompt;
  for (const auto &msg : request.messages) {
    prompt += msg.first + ": " + msg.second + "\n";
  }
  prompt += "assistant: ";

  std::vector<llama_token> tokens(prompt.size() + 32);
  int n_tokens = llama_tokenize(model_.get(), prompt.c_str(),
                                static_cast<int>(prompt.size()), tokens.data(),
                                static_cast<int>(tokens.size()), true, true);
  if (n_tokens < 0) {
    tokens.resize(-n_tokens);
    n_tokens = llama_tokenize(model_.get(), prompt.c_str(),
                              static_cast<int>(prompt.size()), tokens.data(),
                              static_cast<int>(tokens.size()), true, true);
  }
  tokens.resize(n_tokens);

  InferenceStats stats;
  stats.prompt_tokens = static_cast<size_t>(n_tokens);

  auto t_prompt_start = std::chrono::steady_clock::now();

  llama_batch batch =
      llama_batch_get_one(tokens.data(), static_cast<int>(tokens.size()));
  if (llama_decode(ctx_.get(), batch) != 0) {
    std::cerr << "Failed to decode prompt" << std::endl;
    return response;
  }

  auto t_prompt_end = std::chrono::steady_clock::now();
  stats.prompt_processing_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_prompt_end -
                                                            t_prompt_start);

  struct llama_sampler *smpl =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl, llama_sampler_init_temp(request.temperature));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(request.top_p, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

  std::string generated_text;
  auto t_gen_start = std::chrono::steady_clock::now();
  bool first_token = true;

  for (int i = 0; i < request.max_tokens; ++i) {
    llama_token id = llama_sampler_sample(smpl, ctx_.get(), -1);
    llama_sampler_accept(smpl, id);

    if (llama_token_is_eog(model_.get(), id))
      break;

    if (first_token) {
      stats.time_to_first_token =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - t_prompt_end);
      first_token = false;
    }

    char buf[256];
    int n = llama_token_to_piece(model_.get(), id, buf, sizeof(buf), 0, true);

    // Dynamic buffer allocation for edge-case tokens exceeding 256 bytes
    if (n > 0) {
      generated_text.append(buf, n);
    } else if (n < 0) {
      std::vector<char> large_buf(-n);
      llama_token_to_piece(model_.get(), id, large_buf.data(), large_buf.size(),
                           0, true);
      generated_text.append(large_buf.data(), large_buf.size());
    }

    stats.generated_tokens++;

    batch = llama_batch_get_one(&id, 1);
    if (llama_decode(ctx_.get(), batch) != 0)
      break;
  }

  auto t_gen_end = std::chrono::steady_clock::now();
  stats.total_inference_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_gen_end -
                                                            t_gen_start);
  stats.peak_memory_bytes = get_peak_memory();

  llama_sampler_free(smpl);

  ChatChoice choice;
  choice.index = 0;
  choice.message = generated_text;
  choice.finish_reason = "stop";
  response.choices.push_back(choice);
  response.stats = stats;

  return response;
}

void InferenceEngine::chat_completion_stream(
    const ChatRequest &request,
    std::function<void(const std::string &, bool, bool, const InferenceStats &)>
        callback) {
  // Protect context access internally. Note: If network I/O in the callback is
  // slow, it will still block the engine queue.
  std::lock_guard<std::mutex> lock(mutex_);

  if (!is_loaded()) {
    InferenceStats empty;
    callback("", false, true, empty);
    return;
  }

  llama_kv_cache_clear(ctx_.get());

  std::string prompt;
  for (const auto &msg : request.messages) {
    prompt += msg.first + ": " + msg.second + "\n";
  }
  prompt += "assistant: ";

  std::vector<llama_token> tokens(prompt.size() + 32);
  int n_tokens = llama_tokenize(model_.get(), prompt.c_str(),
                                static_cast<int>(prompt.size()), tokens.data(),
                                static_cast<int>(tokens.size()), true, true);
  if (n_tokens < 0) {
    tokens.resize(-n_tokens);
    n_tokens = llama_tokenize(model_.get(), prompt.c_str(),
                              static_cast<int>(prompt.size()), tokens.data(),
                              static_cast<int>(tokens.size()), true, true);
  }
  tokens.resize(n_tokens);

  InferenceStats stats;
  stats.prompt_tokens = static_cast<size_t>(n_tokens);

  auto t_prompt_end = std::chrono::steady_clock::now();

  llama_batch batch =
      llama_batch_get_one(tokens.data(), static_cast<int>(tokens.size()));
  if (llama_decode(ctx_.get(), batch) != 0) {
    callback("", false, true, stats);
    return;
  }
  t_prompt_end = std::chrono::steady_clock::now();

  struct llama_sampler *smpl =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl, llama_sampler_init_temp(request.temperature));
  llama_sampler_chain_add(smpl, llama_sampler_init_top_p(request.top_p, 1));
  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

  auto t_gen_start = std::chrono::steady_clock::now();
  bool is_first = true;

  for (int i = 0; i < request.max_tokens; ++i) {
    llama_token id = llama_sampler_sample(smpl, ctx_.get(), -1);
    llama_sampler_accept(smpl, id);

    if (llama_token_is_eog(model_.get(), id)) {
      stats.total_inference_time =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - t_gen_start);
      callback("", false, true, stats);
      break;
    }

    if (is_first) {
      stats.time_to_first_token =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - t_prompt_end);
    }

    char buf[256];
    int n = llama_token_to_piece(model_.get(), id, buf, sizeof(buf), 0, true);

    std::string piece;
    if (n > 0) {
      piece = std::string(buf, n);
    } else if (n < 0) {
      std::vector<char> large_buf(-n);
      llama_token_to_piece(model_.get(), id, large_buf.data(), large_buf.size(),
                           0, true);
      piece = std::string(large_buf.data(), large_buf.size());
    }

    stats.generated_tokens++;
    callback(piece, is_first, false, stats);

    is_first = false;

    batch = llama_batch_get_one(&id, 1);
    if (llama_decode(ctx_.get(), batch) != 0) {
      stats.total_inference_time =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - t_gen_start);
      callback("", false, true, stats);
      break;
    }
  }

  llama_sampler_free(smpl);
}

} // namespace arm_ai
