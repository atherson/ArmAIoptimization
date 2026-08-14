#include "inference_engine.hpp"

#include <llama.h>
#include <ggml.h>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace arm_ai {

InferenceEngine::InferenceEngine() = default;

InferenceEngine::~InferenceEngine() {
    cleanup();
}

void InferenceEngine::cleanup() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    ctx_.reset();
    model_.reset();
}

bool InferenceEngine::load_model(const std::string& model_path, int n_ctx, int n_threads, int n_gpu_layers) {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanup();

    model_path_ = model_path;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    struct llama_model* model_ptr = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_ptr) {
        std::cerr << "InferenceEngine: Failed to load model from " << model_path << std::endl;
        return false;
    }
    model_ = std::unique_ptr<llama_model, decltype(&llama_model_free)>(model_ptr, &llama_model_free);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = n_threads > 0 ? n_threads : std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;

    struct llama_context* ctx_ptr = llama_new_context_with_model(model_.get(), ctx_params);
    if (!ctx_ptr) {
        std::cerr << "InferenceEngine: Failed to create context." << std::endl;
        return false;
    }
    ctx_ = std::unique_ptr<llama_context, decltype(&llama_free)>(ctx_ptr, &llama_free);

    // Initialize default sampler chain
    auto sparams = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());

    return true;
}

bool InferenceEngine::is_loaded() const {
    return model_ != nullptr && ctx_ != nullptr;
}

std::string InferenceEngine::get_model_name() const {
    if (!model_) return "";
    char buf[256];
    int len = llama_model_desc(model_.get(), buf, sizeof(buf));
    return len > 0 ? std::string(buf, len) : "unknown";
}

size_t InferenceEngine::get_peak_memory() const {
    if (!ctx_) return 0;
    // Return approximate memory usage estimation or state tracking
    return llama_get_model_size(model_.get());
}

ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    ChatResponse response;
    if (!is_loaded()) return response;

    // Clear KV cache using modern seq_rm API
    llama_kv_cache_seq_rm(ctx_.get(), -1, 0, -1);

    const auto* vocab = llama_model_get_vocab(model_.get());

    // Format prompt/messages
    std::string prompt = "";
    for (const auto& msg : request.messages) {
        prompt += msg.role + ": " + msg.content + "\n";
    }
    prompt += "assistant: ";

    std::vector<llama_token> tokens(prompt.size() + 3);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    ChatResponseStats stats;
    stats.prompt_tokens = n_tokens;

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_.get(), batch) != 0) {
        std::cerr << "InferenceEngine: Failed to decode prompt batch." << std::endl;
        return response;
    }

    int generated = 0;
    std::string completion_text = "";
    auto start_time = std::chrono::high_resolution_clock::now();

    while (generated < request.max_tokens) {
        llama_token id = llama_sampler_sample(sampler_, ctx_.get(), -1);
        
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            completion_text.append(buf, n);
        }

        generated++;
        stats.generated_tokens++;

        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_.get(), batch) != 0) {
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    stats.total_time_ms = elapsed.count() * 1000.0;

    ChatChoice choice;
    choice.message = completion_text;
    response.choices.push_back(choice);
    response.stats = stats;

    return response;
}

void InferenceEngine::chat_completion_stream(
    const ChatRequest& request,
    std::function<void(const std::string&, bool, bool, const InferenceStats&)> callback) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_loaded()) return;

    // Clear KV cache using modern seq_rm API
    llama_kv_cache_seq_rm(ctx_.get(), -1, 0, -1);

    const auto* vocab = llama_model_get_vocab(model_.get());

    std::string prompt = "";
    for (const auto& msg : request.messages) {
        prompt += msg.role + ": " + msg.content + "\n";
    }
    prompt += "assistant: ";

    std::vector<llama_token> tokens(prompt.size() + 3);
    int n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    InferenceStats stats;
    stats.prompt_tokens = n_tokens;

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_.get(), batch) != 0) {
        return;
    }

    int generated = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (generated < request.max_tokens) {
        llama_token id = llama_sampler_sample(sampler_, ctx_.get(), -1);

        bool is_eog = llama_vocab_is_eog(vocab, id);
        if (is_eog) {
            callback("", generated == 0, true, stats);
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        std::string piece = "";
        if (n > 0) {
            piece = std::string(buf, n);
        }

        generated++;
        stats.generated_tokens++;

        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        stats.total_time_ms = elapsed.count() * 1000.0;

        callback(piece, generated == 1, false, stats);

        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_.get(), batch) != 0) {
            break;
        }
    }
}

} // namespace arm_ai
