#include "inference_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "common.h"  // for common_tokenize, common_detokenize, etc.

namespace arm_ai {


InferenceEngine::InferenceEngine() {
    // Initialize llama backend
    llama_backend_init();
}

InferenceEngine::~InferenceEngine() {
    llama_backend_free();
}

bool InferenceEngine::load_model(const std::string& model_path, int n_ctx, 
                                  int n_threads, int n_gpu_layers) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free any existing model/context
    ctx_.reset();
    model_.reset();
    sampler_.reset();

    n_ctx_ = n_ctx;
    n_threads_ = (n_threads <= 0) ? static_cast<int>(std::thread::hardware_concurrency()) : n_threads;

    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    // Load model
    model_.reset(llama_model_load_from_file(model_path.c_str(), model_params));
    if (!model_) {
        return false;
    }

    // Get model name from metadata
    char name_buf[256] = {0};
    if (llama_model_meta_val_str(model_.get(), "general.name", name_buf, sizeof(name_buf)) > 0) {
        model_name_ = name_buf;
    } else {
        model_name_ = model_path.substr(model_path.find_last_of("/\\") + 1);
    }

    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_threads = n_threads_;
    ctx_params.n_threads_batch = n_threads_;
    ctx_params.embeddings = false;
    ctx_params.logits_all = false;

    // Create context
    ctx_.reset(llama_new_context_with_model(model_.get(), ctx_params));
    if (!ctx_) {
        model_.reset();
        return false;
    }

    // Create sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_.reset(llama_sampler_chain_init(sampler_params));

    // Add sampling chain: temperature -> top_k -> top_p -> typical -> min_p
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_typical(1.0f, 1));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Reset peak memory tracking
    peak_memory_ = 0;

    return true;
}


std::vector<llama_token> InferenceEngine::tokenize(const std::string& text, bool add_bos) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!model_) {
        return {};
    }

    // Use common_tokenize from llama.cpp's common.h
    std::vector<llama_token> tokens;
    tokens.reserve(text.length() / 2);
    tokens = common_tokenize(model_.get(), text, add_bos);
    return tokens;
}

std::string InferenceEngine::detokenize(const std::vector<llama_token>& tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!ctx_) {
        return "";
    }

    std::string result;
    result.reserve(tokens.size() * 4);
    // Use llama_detokenize (or common_detokenize)
    result = common_detokenize(ctx_.get(), tokens);
    return result;
}


std::string InferenceEngine::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages) {
    
    std::string prompt;
    for (const auto& [role, content] : messages) {
        if (role == "system") {
            prompt += "<|start_header_id|>system<|end_header_id|>\n\n" + content + "<|eot_id|>";
        } else if (role == "user") {
            prompt += "<|start_header_id|>user<|end_header_id|>\n\n" + content + "<|eot_id|>";
        } else if (role == "assistant") {
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" + content + "<|eot_id|>";
        }
    }
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return prompt;
}

llama_batch InferenceEngine::build_batch(const std::vector<llama_token>& tokens, 
                                          int n_past, int seq_id) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = n_past + i;
        batch.seq_id[i] = seq_id;
        batch.logits[i] = (i == tokens.size() - 1);  // Only compute logits for the last token
    }
    batch.n_tokens = tokens.size();
    
    return batch;
}


llama_token InferenceEngine::sample_token(float temperature, float top_p, float top_k) {
    if (!sampler_) {
        return llama_token_nl;  // fallback
    }

    // Reset sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler* new_sampler = llama_sampler_chain_init(sampler_params);
    
    llama_sampler_chain_add(new_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_top_k(static_cast<int>(top_k)));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_typical(1.0f, 1));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(new_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    
    sampler_.reset(new_sampler);
    
    // Sample
    llama_token token = llama_sampler_sample(sampler_.get(), ctx_.get(), -1);
    llama_sampler_accept(sampler_.get(), token);
    
    return token;
}


ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    auto start_total = std::chrono::steady_clock::now();
    InferenceStats stats;
    
    std::lock_guard<std::mutex> lock(mutex_);

    if (!model_ || !ctx_) {
        throw std::runtime_error("Model not loaded");
    }

    // 1. Apply chat template
    std::string prompt = apply_chat_template(request.messages);
    
    // 2. Tokenize
    auto start_prompt = std::chrono::steady_clock::now();
    std::vector<llama_token> prompt_tokens = tokenize(prompt, true);
    stats.prompt_tokens = prompt_tokens.size();
    auto end_prompt = std::chrono::steady_clock::now();
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prompt - start_prompt);

    // 3. Process prompt (prefill)
    // Batch the prompt tokens and run llama_decode
    llama_batch batch = build_batch(prompt_tokens, 0, 0);
    
    auto start_prefill = std::chrono::steady_clock::now();
    if (llama_decode(ctx_.get(), batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Failed to process prompt");
    }
    auto end_prefill = std::chrono::steady_clock::now();
    
    // Update prompt processing time to include actual decode
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prefill - start_prefill);
    
    llama_batch_free(batch);

    // 4. Generate tokens
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(request.max_tokens);
    
    int n_past = prompt_tokens.size();
    std::string generated_text;
    bool first_token = true;
    auto first_token_time = std::chrono::steady_clock::now();

    for (int i = 0; i < request.max_tokens; i++) {
        // Sample next token
        llama_token token = sample_token(request.temperature, request.top_p);
        
        if (token == llama_token_eos(model_.get()) || token == llama_token_nl) {
            // For some models, NL is used as stop
            break;
        }
        
        generated_tokens.push_back(token);
        
        // Detokenize incrementally (or at the end)
        if (first_token) {
            first_token_time = std::chrono::steady_clock::now();
            stats.time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(
                first_token_time - start_prefill);
            first_token = false;
        }
        
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

    // Detokenize all generated tokens
    generated_text = detokenize(generated_tokens);

    // Build response
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

    // Update peak memory
    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());

    return response;
}


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

    // 1. Apply chat template
    std::string prompt = apply_chat_template(request.messages);
    
    // 2. Tokenize
    auto start_prompt = std::chrono::steady_clock::now();
    std::vector<llama_token> prompt_tokens = tokenize(prompt, true);
    stats.prompt_tokens = prompt_tokens.size();
    auto end_prompt = std::chrono::steady_clock::now();
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_prompt - start_prompt);

    // 3. Process prompt (prefill)
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

    // 4. Generate tokens with streaming
    int n_past = prompt_tokens.size();
    bool first_token = true;
    auto first_token_time = std::chrono::steady_clock::now();
    std::string accumulated_text;
    std::vector<llama_token> generated_tokens;

    for (int i = 0; i < request.max_tokens; i++) {
        // Sample next token
        llama_token token = sample_token(request.temperature, request.top_p);
        
        if (token == llama_token_eos(model_.get())) {
            break;
        }
        
        generated_tokens.push_back(token);
        
        // Detokenize incrementally
        std::string token_text = detokenize({token});
        
        if (first_token) {
            first_token_time = std::chrono::steady_clock::now();
            stats.time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(
                first_token_time - start_prefill);
            first_token = false;
        }
        
        accumulated_text += token_text;
        stats.generated_tokens = generated_tokens.size();
        stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_prefill);
        
        // Callback with the new token
        callback(token_text, (i == 0), false, stats);
        
        // Update n_past and decode next token
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
    
    // Update peak memory
    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());

    // Final callback
    callback("", false, true, stats);
}


size_t InferenceEngine::get_peak_memory() const {
  
    if (ctx_) {

        return llama_model_size(model_.get()) + llama_context_size(ctx_.get());
    }
    return 0;
}

} 
