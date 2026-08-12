#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>

#include "llama.h"

namespace arm_ai {

struct InferenceStats {
    std::chrono::microseconds prompt_processing_time{0};
    std::chrono::microseconds time_to_first_token{0};
    std::chrono::microseconds total_inference_time{0};
    size_t prompt_tokens = 0;
    size_t generated_tokens = 0;
    size_t peak_memory_bytes = 0;
    
    double tokens_per_second() const {
        if (total_inference_time.count() == 0) return 0.0;
        return static_cast<double>(generated_tokens) * 1'000'000.0 / 
               static_cast<double>(total_inference_time.count());
    }
    
    double prompt_processing_tokens_per_second() const {
        if (prompt_processing_time.count() == 0) return 0.0;
        return static_cast<double>(prompt_tokens) * 1'000'000.0 / 
               static_cast<double>(prompt_processing_time.count());
    }
};

struct ChatRequest {
    std::string model;
    std::vector<std::pair<std::string, std::string>> messages;  // role, content
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

    // Load a model from a GGUF file
    bool load_model(const std::string& model_path, int n_ctx = 4096, 
                    int n_threads = -1, int n_gpu_layers = 0);

    // Check if a model is loaded
    bool is_loaded() const { return model_ != nullptr && ctx_ != nullptr; }

    // Get model name
    std::string get_model_name() const { return model_name_; }

    // Perform chat completion (non-streaming)
    ChatResponse chat_completion(const ChatRequest& request);

    void chat_completion_stream(
        const ChatRequest& request,
        std::function<void(const std::string&, bool, bool, const InferenceStats&)> callback
    );

    // Get peak memory usage in bytes
    size_t get_peak_memory() const;

private:
    // Tokenize a string
    std::vector<llama_token> tokenize(const std::string& text, bool add_bos = true);

    // Detokenize tokens to string
    std::string detokenize(const std::vector<llama_token>& tokens);

    // Apply chat template
    std::string apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages);

    // Build inference batch
    llama_batch build_batch(const std::vector<llama_token>& tokens, 
                            int n_past, int seq_id);

    // Sampling helpers
    llama_token sample_token(float temperature, float top_p, float top_k = 40.0f);

private:
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx_{nullptr, llama_free};
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_{nullptr, llama_sampler_free};

    std::string model_name_;
    int n_ctx_ = 4096;
    int n_threads_ = -1;
    
    mutable std::mutex mutex_;
    std::atomic<size_t> peak_memory_{0};

    // Cache for the last tokenization to avoid re-tokenizing
    std::vector<llama_token> cached_tokens_;
    std::string cached_text_;
};

} 
