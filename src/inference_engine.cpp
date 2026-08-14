#include "inference_engine.hpp"

#include <llama.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>   // for std::thread::hardware_concurrency

namespace arm_ai {

// ---------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------
InferenceEngine::InferenceEngine()
    : model_(nullptr, llama_model_free),
      ctx_(nullptr, llama_free) {}

InferenceEngine::~InferenceEngine() {
    // unique_ptrs automatically free
}

// ---------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------
bool InferenceEngine::load_model(const std::string& model_path,
                                 int n_ctx,
                                 int n_threads,
                                 int n_gpu_layers) {
    if (is_loaded()) return true;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    model_.reset(llama_model_load_from_file(model_path.c_str(), model_params));
    if (!model_) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return false;
    }

    // Get vocabulary (needed for tokenization)
    vocab_ = llama_model_get_vocab(model_.get());
    if (!vocab_) {
        std::cerr << "Failed to get vocabulary" << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = (n_threads > 0) ? n_threads : std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;

    // Note: llama_new_context_with_model is deprecated but still works.
    // For newer versions, use llama_init_from_model (but that requires different params).
    ctx_.reset(llama_new_context_with_model(model_.get(), ctx_params));
    if (!ctx_) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }

    n_ctx_ = n_ctx;
    n_threads_ = ctx_params.n_threads;
    model_name_ = model_path;

    return true;
}

// ---------------------------------------------------------------------
// Tokenization / Detokenization (now using vocab)
// ---------------------------------------------------------------------
std::vector<llama_token> InferenceEngine::tokenize(const std::string& text, bool add_bos) {
    if (!vocab_) return {};

    std::vector<llama_token> tokens;
    const int n_tokens = llama_tokenize(vocab_, text.c_str(), text.size(),
                                        tokens.data(), tokens.size(), add_bos, false);
    if (n_tokens < 0) {
        std::cerr << "Failed to tokenize" << std::endl;
        return {};
    }
    tokens.resize(n_tokens);
    llama_tokenize(vocab_, text.c_str(), text.size(),
                   tokens.data(), tokens.size(), add_bos, false);
    return tokens;
}

std::string InferenceEngine::detokenize(const std::vector<llama_token>& tokens) {
    if (!vocab_) return "";

    std::string result;
    for (const auto& token : tokens) {
        std::string piece = llama_token_to_piece(vocab_, token);
        result += piece;
    }
    return result;
}

// ---------------------------------------------------------------------
// Chat template (unchanged)
// ---------------------------------------------------------------------
std::string InferenceEngine::apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages) {
    if (!model_) return "";

    std::vector<llama_chat_message> msgs;
    msgs.reserve(messages.size());
    for (const auto& [role, content] : messages) {
        msgs.push_back({role.c_str(), content.c_str()});
    }

    // Pass nullptr for template to auto-detect from model metadata
    int n = llama_chat_apply_template(nullptr, msgs.data(), msgs.size(), true, nullptr, 0);
    if (n < 0) return "";
    std::string buf(n + 1, '\0');
    n = llama_chat_apply_template(nullptr, msgs.data(), msgs.size(), true, buf.data(), buf.size());
    if (n < 0) return "";
    buf.resize(n);
    return buf;
}

// ---------------------------------------------------------------------
// Build a batch (unchanged)
// ---------------------------------------------------------------------
llama_batch InferenceEngine::build_batch(const std::vector<llama_token>& tokens,
                                         int n_past,
                                         int seq_id) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = n_past + i;
        batch.seq_id[i][0] = seq_id;
        batch.n_seq_id[i] = 1;
        batch.logits[i] = (i == tokens.size() - 1);
    }
    batch.n_tokens = tokens.size();
    return batch;
}

// ---------------------------------------------------------------------
// Sampling – using new sampler chain API
// ---------------------------------------------------------------------
llama_token InferenceEngine::sample_token(llama_sampler* sampler) {
    return llama_sampler_sample(sampler, ctx_.get(), -1);
}

// Helper to create a sampler with temp and top_p
static llama_sampler* make_sampler(float temp, float top_p) {
    // Create a default chain (min_p, temp, top_p, etc.)
    llama_sampler* chain = llama_sampler_chain_default();
    // We can add custom samplers, but the default chain already includes temp and top_p.
    // However we can modify parameters: use llama_sampler_chain_get to get each stage.
    // Simpler: create a custom sampler chain with just softmax + temp + top_p.
    // But for simplicity, we'll use llama_sampler_chain_default which includes all standard.
    // We can then adjust temperature and top_p using set functions if they exist.
    // In current API, we can set via llama_sampler_set_temp and llama_sampler_set_top_p
    // if the sampler supports it. The chain default returns a chain of samplers.
    // Let's create a chain manually:
    llama_sampler* chain = llama_sampler_chain_new();
    llama_sampler_chain_add(chain, llama_sampler_init_softmax());
    // Add temperature and top_p as separate stages (order matters: top_p before temp? Usually temp then top_p)
    // We'll add temp and top_p as samplers:
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1)); // min_keep=1
    return chain;
}

// ---------------------------------------------------------------------
// Main generation (non‑streaming)
// ---------------------------------------------------------------------
ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    ChatResponse response;
    if (!model_ || !ctx_ || !vocab_) {
        std::cerr << "Model not loaded" << std::endl;
        return response;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Build prompt
    std::string prompt = apply_chat_template(request.messages);
    if (prompt.empty()) {
        std::cerr << "Failed to apply chat template" << std::endl;
        return response;
    }

    // Tokenize prompt
    auto tokens = tokenize(prompt, true);
    if (tokens.empty()) {
        std::cerr << "Tokenization failed" << std::endl;
        return response;
    }

    InferenceStats stats;
    stats.prompt_tokens = tokens.size();

    auto start_total = std::chrono::steady_clock::now();
    int n_past = 0;
    llama_batch batch = build_batch(tokens, n_past, 0);
    auto start_prompt = std::chrono::steady_clock::now();
    if (llama_decode(ctx_.get(), batch) != 0) {
        std::cerr << "llama_decode failed" << std::endl;
        llama_batch_free(batch);
        return response;
    }
    auto end_prompt = std::chrono::steady_clock::now();
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(end_prompt - start_prompt);
    n_past += tokens.size();
    llama_batch_free(batch);

    // Create sampler
    llama_sampler* sampler = make_sampler(request.temperature, request.top_p);
    if (!sampler) {
        std::cerr << "Failed to create sampler" << std::endl;
        return response;
    }

    std::vector<llama_token> generated_tokens;
    bool first_token = true;
    int remaining = request.max_tokens;

    while (remaining-- > 0) {
        llama_token id = llama_sampler_sample(sampler, ctx_.get(), -1);
        if (llama_vocab_is_eog(vocab_, id)) break;

        generated_tokens.push_back(id);

        llama_batch single = llama_batch_init(1, 0, 1);
        single.token[0] = id;
        single.pos[0] = n_past;
        single.seq_id[0][0] = 0;
        single.n_seq_id[0] = 1;
        single.logits[0] = true;
        single.n_tokens = 1;

        if (llama_decode(ctx_.get(), single) != 0) {
            std::cerr << "llama_decode failed during generation" << std::endl;
            llama_batch_free(single);
            break;
        }
        n_past++;
        llama_batch_free(single);

        if (first_token) {
            stats.time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_prompt);
            first_token = false;
        }
    }

    llama_sampler_free(sampler);

    auto end_total = std::chrono::steady_clock::now();
    stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
    stats.generated_tokens = generated_tokens.size();

    response.id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    response.model = model_name_;
    response.choices.emplace_back(ChatChoice{0, detokenize(generated_tokens), "stop"});
    response.stats = stats;
    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());

    return response;
}

// ---------------------------------------------------------------------
// Streaming generation
// ---------------------------------------------------------------------
void InferenceEngine::chat_completion_stream(
    const ChatRequest& request,
    std::function<void(const std::string&, bool, bool, const InferenceStats&)> callback) {
    if (!model_ || !ctx_ || !vocab_) {
        std::cerr << "Model not loaded" << std::endl;
        callback("", false, true, {});
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::string prompt = apply_chat_template(request.messages);
    if (prompt.empty()) {
        std::cerr << "Failed to apply chat template" << std::endl;
        callback("", false, true, {});
        return;
    }

    auto tokens = tokenize(prompt, true);
    if (tokens.empty()) {
        std::cerr << "Tokenization failed" << std::endl;
        callback("", false, true, {});
        return;
    }

    InferenceStats stats;
    stats.prompt_tokens = tokens.size();

    auto start_total = std::chrono::steady_clock::now();
    int n_past = 0;
    llama_batch batch = build_batch(tokens, n_past, 0);
    auto start_prompt = std::chrono::steady_clock::now();
    if (llama_decode(ctx_.get(), batch) != 0) {
        std::cerr << "llama_decode failed" << std::endl;
        llama_batch_free(batch);
        callback("", false, true, {});
        return;
    }
    auto end_prompt = std::chrono::steady_clock::now();
    stats.prompt_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(end_prompt - start_prompt);
    n_past += tokens.size();
    llama_batch_free(batch);

    llama_sampler* sampler = make_sampler(request.temperature, request.top_p);
    if (!sampler) {
        std::cerr << "Failed to create sampler" << std::endl;
        callback("", false, true, {});
        return;
    }

    bool first_token = true;
    int remaining = request.max_tokens;

    while (remaining-- > 0) {
        llama_token id = llama_sampler_sample(sampler, ctx_.get(), -1);
        if (llama_vocab_is_eog(vocab_, id)) break;

        std::string piece = llama_token_to_piece(vocab_, id);
        if (first_token) {
            stats.time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_prompt);
            callback(piece, true, false, stats);
            first_token = false;
        } else {
            callback(piece, false, false, stats);
        }

        llama_batch single = llama_batch_init(1, 0, 1);
        single.token[0] = id;
        single.pos[0] = n_past;
        single.seq_id[0][0] = 0;
        single.n_seq_id[0] = 1;
        single.logits[0] = true;
        single.n_tokens = 1;

        if (llama_decode(ctx_.get(), single) != 0) {
            std::cerr << "llama_decode failed during streaming" << std::endl;
            llama_batch_free(single);
            break;
        }
        n_past++;
        llama_batch_free(single);
    }

    llama_sampler_free(sampler);

    auto end_total = std::chrono::steady_clock::now();
    stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
    stats.generated_tokens = n_past - tokens.size(); // approximate
    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());

    callback("", false, true, stats);
}

// ---------------------------------------------------------------------
// Peak memory tracking (placeholder)
// ---------------------------------------------------------------------
size_t InferenceEngine::get_peak_memory() const {
    return 0; // Could be extended with OS-specific calls
}

} // namespace arm_ai
