#include "inference_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#include "llama.h"  

namespace arm_ai {

// RAII helpers for llama types

InferenceEngine::InferenceEngine() {
    llama_backend_init();
}

InferenceEngine::~InferenceEngine() {
    llama_backend_free();
}

// Model loading

bool InferenceEngine::load_model(const std::string& model_path, int n_ctx,
                                  int n_threads, int n_gpu_layers) {
    std::lock_guard<std::mutex> lock(mutex_);

    ctx_.reset();
    model_.reset();
    sampler_.reset();

    n_ctx_ = n_ctx;
    n_threads_ = (n_threads <= 0) ? static_cast<int>(std::thread::hardware_concurrency()) : n_threads;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    model_.reset(llama_model_load_from_file(model_path.c_str(), model_params));
    if (!model_) {
        return false;
    }

    char name_buf[256] = {0};
    if (llama_model_meta_val_str(model_.get(), "general.name", name_buf, sizeof(name_buf)) > 0) {
        model_name_ = name_buf;
    } else {
        model_name_ = model_path.substr(model_path.find_last_of("/\\") + 1);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_threads = n_threads_;
    ctx_params.n_threads_batch = n_threads_;
    ctx_params.embeddings = false;
    ctx_params.logits_all = false;

    ctx_.reset(llama_new_context_with_model(model_.get(), ctx_params));
    if (!ctx_) {
        model_.reset();
        return false;
    }

    // sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_.reset(llama_sampler_chain_init(sampler_params));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_typical(1.0f, 1));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    peak_memory_ = 0;
    return true;
}

// Tokenization using public API

std::vector<llama_token> InferenceEngine::tokenize(const std::string& text, bool add_bos) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_) return {};

    std::vector<llama_token> tokens(text.length() + 1);  // reserve
    int n_tokens = llama_tokenize(model_.get(), text.c_str(), text.length(),
                                  tokens.data(), tokens.size(), add_bos, false);
    if (n_tokens < 0) return {};
    tokens.resize(n_tokens);
    return tokens;
}

std::string InferenceEngine::detokenize(const std::vector<llama_token>& tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ctx_) return "";

    std::string result;
    result.reserve(tokens.size() * 4);
    for (llama_token token : tokens) {
        char piece[32];
        int n = llama_token_to_piece(ctx_.get(), token, piece, sizeof(piece), 0, true);
        if (n < 0) continue;
        result.append(piece, n);
    }
    return result;
}

// Chat template using public API

std::string InferenceEngine::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages) {

    // Convert to the format expected by llama_chat_apply_template
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    // We need to store strings to keep pointers valid
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    for (const auto& [role, content] : messages) {
        roles.push_back(role);
        contents.push_back(content);
        chat.push_back({roles.back().c_str(), contents.back().c_str()});
    }

    // Get the required buffer size
    int length = llama_chat_apply_template(model_.get(), nullptr, chat.data(), chat.size(),
                                           false, nullptr, 0);
    if (length < 0) return "";

    std::string result(length, '\0');
    int written = llama_chat_apply_template(model_.get(), nullptr, chat.data(), chat.size(),
                                            false, result.data(), result.size());
    if (written < 0) return "";
    result.resize(written);
    return result;
}

// Batch building

llama_batch InferenceEngine::build_batch(const std::vector<llama_token>& tokens,
                                          int n_past, int seq_id) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = n_past + i;
        batch.seq_id[i] = seq_id;
        batch.logits[i] = (i == tokens.size() - 1);
    }
    batch.n_tokens = tokens.size();
    return batch;
}

// Sampling

llama_token InferenceEngine::sample_token(float temperature, float top_p, float top_k) {
    if (!sampler_) return llama_token_nl;

    // Update sampler parameters (re‑create)
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler* new_sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(new_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_top_k(static_cast<int>(top_k)));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_typical(1.0f, 1));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    sampler_.reset(new_sampler);

    llama_token token = llama_sampler_sample(sampler_.get(), ctx_.get(), -1);
    llama_sampler_accept(sampler_.get(), token);
    return token;
}

// Chat completion

ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    auto start_total = std::chrono::steady_clock::now();
    InferenceStats stats;

    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ || !ctx_) {
        throw std::runtime_error("Model not loaded");
    }

    // 1. Build prompt via chat template
    std::string prompt = apply_chat_template(request.messages);

    // 2. Tokenize
    auto start_prompt = std::chrono::steady_clock::now();
    std::vector<llama_token> prompt_tokens = tokenize(prompt, true);
    stats.prompt_tokens = prompt_tokens.size();
    auto end_prompt = std::chrono::steady_clock::now();
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prompt - start_prompt);

    // 3. Prefill
    llama_batch batch = build_batch(prompt_tokens, 0, 0);
    auto start_prefill = std::chrono::steady_clock::now();
    if (llama_decode(ctx_.get(), batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Failed to process prompt");
    }
    auto end_prefill = std::chrono::steady_clock::now();
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prefill - start_prefill);
    llama_batch_free(batch);

    // 4. Generation loop
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(request.max_tokens);
    int n_past = prompt_tokens.size();
    bool first_token = true;
    auto first_token_time = std::chrono::steady_clock::now();

    for (int i = 0; i < request.max_tokens; i++) {
        llama_token token = sample_token(request.temperature, request.top_p);
        if (token == llama_token_eos(model_.get())) break;

        generated_tokens.push_back(token);
        if (first_token) {
            first_token_time = std::chrono::steady_clock::now();
            stats.time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(
                first_token_time - start_prefill);
            first_token = false;
        }

        // Decode next token
        std::vector<llama_token> single_token = {token};
        llama_batch next_batch = build_batch(single_token, n_past, 0);
        if (llama_decode(ctx_.get(), next_batch) != 0) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
        n_past += 1;
    }

    auto end_total = std::chrono::steady_clock::now();
    stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_total - start_prefill);
    stats.generated_tokens = generated_tokens.size();

    std::string generated_text = detokenize(generated_tokens);

    ChatResponse response;
    response.id = "chatcmpl-" + std::to_string(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    response.model = model_name_;
    response.stats = stats;

    ChatChoice choice;
    choice.index = 0;
    choice.message = generated_text;
    choice.finish_reason = (generated_tokens.size() >= static_cast<size_t>(request.max_tokens))
                           ? "length" : "stop";
    response.choices.push_back(choice);

    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());
    return response;
}

// Streaming

void InferenceEngine::chat_completion_stream(
    const ChatRequest& request,
    std::function<void(const std::string&, bool, bool, const InferenceStats&)> callback) {

    auto start_total = std::chrono::steady_clock::now();
    InferenceStats stats;

    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ || !ctx_) {
        callback("", false, true, stats);
        return;
    }

    std::string prompt = apply_chat_template(request.messages);

    auto start_prompt = std::chrono::steady_clock::now();
    std::vector<llama_token> prompt_tokens = tokenize(prompt, true);
    stats.prompt_tokens = prompt_tokens.size();
    auto end_prompt = std::chrono::steady_clock::now();
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prompt - start_prompt);

    llama_batch batch = build_batch(prompt_tokens, 0, 0);
    auto start_prefill = std::chrono::steady_clock::now();
    if (llama_decode(ctx_.get(), batch) != 0) {
        llama_batch_free(batch);
        callback("", false, true, stats);
        return;
    }
    auto end_prefill = std::chrono::steady_clock::now();
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prefill - start_prefill);
    llama_batch_free(batch);

    int n_past = prompt_tokens.size();
    bool first_token = true;
    auto first_token_time = std::chrono::steady_clock::now();
    std::vector<llama_token> generated_tokens;

    for (int i = 0; i < request.max_tokens; i++) {
        llama_token token = sample_token(request.temperature, request.top_p);
        if (token == llama_token_eos(model_.get())) break;

        generated_tokens.push_back(token);
        std::string token_text = detokenize({token});

        if (first_token) {
            first_token_time = std::chrono::steady_clock::now();
            stats.time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(
                first_token_time - start_prefill);
            first_token = false;
        }

        stats.generated_tokens = generated_tokens.size();
        stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_prefill);
        callback(token_text, (i == 0), false, stats);

        std::vector<llama_token> single_token = {token};
        llama_batch next_batch = build_batch(single_token, n_past, 0);
        if (llama_decode(ctx_.get(), next_batch) != 0) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
        n_past += 1;
    }

    stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_prefill);
    stats.generated_tokens = generated_tokens.size();
    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());
    callback("", false, true, stats);
}

// Memory

size_t InferenceEngine::get_peak_memory() const {
    if (ctx_) {
        return llama_model_size(model_.get()) + llama_context_size(ctx_.get());
    }
    return 0;
}

} // namespace arm_ai
