#include "inference_engine.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace arm_ai {

InferenceEngine::InferenceEngine() {}

InferenceEngine::~InferenceEngine() {
    // Relying on default unique_ptr cleanup for model_ and ctx_
}

bool InferenceEngine::load_model(const std::string& model_path, int context_size, int batch_size, int n_threads) {
    model_name_ = model_path;

    llama_model_params model_params = llama_model_default_params();
    struct llama_model* model_ptr = llama_load_model_from_file(model_path.c_str(), model_params);
    
    if (!model_ptr) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return false;
    }
    
    model_.reset(model_ptr);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size;
    ctx_params.n_batch = batch_size;
    ctx_params.n_threads = n_threads > 0 ? n_threads : std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;

    struct llama_context* ctx_ptr = llama_init_from_model(model_.get(), ctx_params);
    
    if (!ctx_ptr) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    ctx_.reset(ctx_ptr);
    return true;
}

size_t InferenceEngine::get_peak_memory() const {
    if (!model_) return 0;
    return llama_model_size(model_.get());
}

ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    ChatResponse response;
    
    if (!is_loaded()) {
        std::cerr << "Model not loaded" << std::endl;
        return response;
    }

    // Clear KV cache for a new, stateless chat request
    llama_kv_cache_clear(ctx_.get());

    std::string prompt;
    for (const auto& msg : request.messages) {
        prompt += msg.first + ": " + msg.second + "\n";
    }
    prompt += "assistant: ";

    std::vector<llama_token> tokens(prompt.length() + 1);
    int n_tokens = llama_tokenize(model_.get(), prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(model_.get(), prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    InferenceStats stats;
    stats.prompt_tokens = n_tokens;
    stats.generated_tokens = 0;

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_.get(), batch) != 0) {
        std::cerr << "Failed to decode" << std::endl;
        return response;
    }

    // Initialize sampler chain locally for the current generation
    struct llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::string generated_text;
    
    for (int i = 0; i < request.max_tokens; ++i) {
        llama_token id = llama_sampler_sample(smpl, ctx_.get(), -1);
        llama_sampler_accept(smpl, id);
        
        if (llama_token_is_eog(model_.get(), id)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(model_.get(), id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            generated_text += std::string(buf, n);
        }

        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_.get(), batch) != 0) {
            break;
        }
        stats.generated_tokens++;
    }

    // Free the local sampler chain
    llama_sampler_free(smpl);

    response.content = generated_text;
    // Assuming ChatResponse has an InferenceStats member named 'stats' or similar, 
    // assign it here if needed, e.g., response.stats = stats;
    
    return response;
}

void InferenceEngine::chat_completion_stream(const ChatRequest& request, 
                                             std::function<void(const std::string&, bool, bool, const InferenceStats&)> callback) {
    if (!is_loaded()) {
        InferenceStats empty_stats;
        callback("", false, true, empty_stats);
        return;
    }

    llama_kv_cache_clear(ctx_.get());

    std::string prompt;
    for (const auto& msg : request.messages) {
        prompt += msg.first + ": " + msg.second + "\n";
    }
    prompt += "assistant: ";

    std::vector<llama_token> tokens(prompt.length() + 1);
    int n_tokens = llama_tokenize(model_.get(), prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(model_.get(), prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    InferenceStats stats;
    stats.prompt_tokens = n_tokens;
    stats.generated_tokens = 0;

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_.get(), batch) != 0) {
        callback("", false, true, stats);
        return;
    }

    struct llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    for (int i = 0; i < request.max_tokens; ++i) {
        llama_token id = llama_sampler_sample(smpl, ctx_.get(), -1);
        llama_sampler_accept(smpl, id);

        bool is_eog = llama_token_is_eog(model_.get(), id);
        if (is_eog) {
            callback("", false, true, stats);
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(model_.get(), id, buf, sizeof(buf), 0, true);
        std::string piece;
        if (n > 0) {
            piece = std::string(buf, n);
        }

        stats.generated_tokens++;
        callback(piece, true, false, stats);

        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_.get(), batch) != 0) {
            callback("", false, true, stats);
            break;
        }
    }

    llama_sampler_free(smpl);
}

} // namespace arm_ai
