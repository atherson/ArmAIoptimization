#include "inference_engine.hpp"

#include <llama.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

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
    // use_mmap/use_mlock removed in recent API, defaults are fine

    model_.reset(llama_model_load_from_file(model_path.c_str(), model_params));
    if (!model_) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = (n_threads > 0) ? n_threads : std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;

    ctx_.reset(llama_new_context_with_model(model_.get(), ctx_params));
    if (!ctx_) {
        std::cerr << "Failed to create context for model: " << model_path << std::endl;
        return false;
    }

    n_ctx_ = n_ctx;
    n_threads_ = ctx_params.n_threads;
    model_name_ = model_path;

    // Get vocabulary
    const llama_vocab* vocab = llama_model_get_vocab(model_.get());
    if (!vocab) {
        std::cerr << "Failed to get vocabulary" << std::endl;
        return false;
    }

    // Store vocab if needed (but we use model_->vocab in tokenize)

    return true;
}

// ---------------------------------------------------------------------
// Tokenization / Detokenization
// ---------------------------------------------------------------------
std::vector<llama_token> InferenceEngine::tokenize(const std::string& text, bool add_bos) {
    std::vector<llama_token> tokens;
    const int n_tokens = llama_tokenize(model_.get(), text.c_str(), text.size(),
                                        tokens.data(), tokens.size(), add_bos, false);
    if (n_tokens < 0) {
        std::cerr << "Failed to tokenize" << std::endl;
        return {};
    }
    tokens.resize(n_tokens);
    // Re-tokenize to fill
    llama_tokenize(model_.get(), text.c_str(), text.size(),
                   tokens.data(), tokens.size(), add_bos, false);
    return tokens;
}

std::string InferenceEngine::detokenize(const std::vector<llama_token>& tokens) {
    std::string result;
    for (const auto& token : tokens) {
        std::string piece = llama_token_to_piece(model_.get(), token);
        result += piece;
    }
    return result;
}

// ---------------------------------------------------------------------
// Chat template (calls llama_chat_apply_template)
// ---------------------------------------------------------------------
std::string InferenceEngine::apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages) {
    if (!model_) return "";

    std::vector<llama_chat_message> msgs;
    msgs.reserve(messages.size());
    for (const auto& [role, content] : messages) {
        msgs.push_back({role.c_str(), content.c_str()});
    }

    int n = llama_chat_apply_template(nullptr, msgs.data(), msgs.size(), true, nullptr, 0);
    if (n < 0) return "";
    std::string buf(n + 1, '\0');
    n = llama_chat_apply_template(nullptr, msgs.data(), msgs.size(), true, buf.data(), buf.size());
    if (n < 0) return "";
    buf.resize(n);
    return buf;
}

// ---------------------------------------------------------------------
// Build a batch
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
        batch.logits[i] = (i == tokens.size() - 1); // only compute logits for last token
    }
    batch.n_tokens = tokens.size();
    return batch;
}

// ---------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------
llama_token InferenceEngine::sample_token(llama_sampler* sampler) {
    return llama_sampler_sample(sampler, ctx_.get(), -1);
}

// ---------------------------------------------------------------------
// Main generation (non‑streaming)
// ---------------------------------------------------------------------
ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    ChatResponse response;
    if (!model_ || !ctx_) {
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

    // Statistics
    InferenceStats stats;
    stats.prompt_tokens = tokens.size();

    // Prepare batch and decode
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

    // Sampling
    llama_sampler* sampler = llama_sampler_init_softmax();
    // Set temperature, top_p if needed (defaults from request)
    llama_sampler_set_temp(sampler, request.temperature);
    llama_sampler_set_top_p(sampler, request.top_p);

    std::vector<llama_token> generated_tokens;
    bool first_token = true;
    int remaining = request.max_tokens;

    while (remaining-- > 0) {
        // Sample next token
        llama_token id = llama_sampler_sample(sampler, ctx_.get(), -1);
        if (llama_token_is_eog(model_.get(), id)) break;

        generated_tokens.push_back(id);

        // Build batch for this single token (append)
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

    // Build response
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
    if (!model_ || !ctx_) {
        std::cerr << "Model not loaded" << std::endl;
        callback("", false, true, {});
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Build prompt
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

    llama_sampler* sampler = llama_sampler_init_softmax();
    llama_sampler_set_temp(sampler, request.temperature);
    llama_sampler_set_top_p(sampler, request.top_p);

    bool first_token = true;
    int remaining = request.max_tokens;

    while (remaining-- > 0) {
        llama_token id = llama_sampler_sample(sampler, ctx_.get(), -1);
        if (llama_token_is_eog(model_.get(), id)) break;

        std::string piece = llama_token_to_piece(model_.get(), id);
        // Callback with the piece, is_first, is_final=false
        if (first_token) {
            stats.time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_prompt);
            callback(piece, true, false, stats);
            first_token = false;
        } else {
            callback(piece, false, false, stats);
        }

        // Decode the token to update context
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

    // Final callback
    auto end_total = std::chrono::steady_clock::now();
    stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
    stats.generated_tokens = n_past - tokens.size(); // approximate
    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());

    callback("", false, true, stats);
}

// ---------------------------------------------------------------------
// Peak memory tracking
// ---------------------------------------------------------------------
size_t InferenceEngine::get_peak_memory() const {
    // Use llama_get_memory_usage if available, else fallback to 0
    // llm.cpp may not expose, so we use a simple placeholder
    // In practice, you can call llama_ctx_memory_usage(ctx_.get())
    // but that might not be public. We'll return a dummy.
    // For real tracking, you can implement with OS-specific calls.
    return 0; // Placeholder
}

} // namespace arm_ai
