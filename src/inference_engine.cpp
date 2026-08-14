#include "inference_engine.hpp"

#include <llama.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>

namespace arm_ai {

// ---------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------
InferenceEngine::InferenceEngine()
    : model_(nullptr, llama_model_free),
      ctx_(nullptr, llama_free) {}

InferenceEngine::~InferenceEngine() = default;

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

    vocab_ = llama_model_get_vocab(model_.get());
    if (!vocab_) {
        std::cerr << "Failed to get vocabulary" << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = (n_threads > 0) ? n_threads : std::thread::hardware_concurrency();
    ctx_params.n_threads_batch = ctx_params.n_threads;

    // llama_new_context_with_model is deprecated but still works.
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
// Tokenization / Detokenization (using vocab)
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
    std::vector<char> buffer(16); // initial size

    for (const auto& token : tokens) {
        int n = llama_token_to_piece(vocab_, token, buffer.data(), buffer.size(), 0, true);
        if (n < 0) {
            buffer.resize(-n + 1);
            n = llama_token_to_piece(vocab_, token, buffer.data(), buffer.size(), 0, true);
        }
        if (n > 0) {
            result.append(buffer.data(), n);
        }
    }
    return result;
}

// ---------------------------------------------------------------------
// Chat template
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
// Build batch
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
// Non‑streaming generation (uses legacy sampling)
// ---------------------------------------------------------------------
ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    ChatResponse response;
    if (!model_ || !ctx_ || !vocab_) {
        std::cerr << "Model not loaded" << std::endl;
        return response;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::string prompt = apply_chat_template(request.messages);
    if (prompt.empty()) {
        std::cerr << "Failed to apply chat template" << std::endl;
        return response;
    }

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

    // Prepare sampling parameters
    llama_sampling_params sparams;
    sparams.temp = request.temperature;
    sparams.top_p = request.top_p;
    sparams.penalty_last_n = 64;        // optional
    sparams.penalty_repeat = 1.0f;      // no penalty
    sparams.penalty_freq = 0.0f;
    sparams.penalty_present = 0.0f;
    sparams.mirostat = 0;               // disabled
    sparams.mirostat_tau = 5.0f;
    sparams.mirostat_eta = 0.1f;

    std::vector<llama_token> generated_tokens;
    bool first_token = true;
    int remaining = request.max_tokens;

    while (remaining-- > 0) {
        llama_token id = llama_sample_token(ctx_.get(), sparams);
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

    auto end_total = std::chrono::steady_clock::now();
    stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
    stats.generated_tokens = generated_tokens.size();

    response.id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    response.model = model_name_;
    if (!generated_tokens.empty()) {
        response.choices.emplace_back(ChatChoice{0, detokenize(generated_tokens), "stop"});
    } else {
        response.choices.emplace_back(ChatChoice{0, "", "stop"});
    }
    response.stats = stats;
    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());

    return response;
}

// ---------------------------------------------------------------------
// Streaming generation (uses legacy sampling)
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

    // Sampling parameters
    llama_sampling_params sparams;
    sparams.temp = request.temperature;
    sparams.top_p = request.top_p;
    sparams.penalty_last_n = 64;
    sparams.penalty_repeat = 1.0f;
    sparams.penalty_freq = 0.0f;
    sparams.penalty_present = 0.0f;
    sparams.mirostat = 0;
    sparams.mirostat_tau = 5.0f;
    sparams.mirostat_eta = 0.1f;

    bool first_token = true;
    int remaining = request.max_tokens;

    std::vector<char> buffer(16); // for token-to-piece

    while (remaining-- > 0) {
        llama_token id = llama_sample_token(ctx_.get(), sparams);
        if (llama_vocab_is_eog(vocab_, id)) break;

        // Convert token to string
        int n = llama_token_to_piece(vocab_, id, buffer.data(), buffer.size(), 0, true);
        if (n < 0) {
            buffer.resize(-n + 1);
            n = llama_token_to_piece(vocab_, id, buffer.data(), buffer.size(), 0, true);
        }
        std::string piece(buffer.data(), n);

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

    auto end_total = std::chrono::steady_clock::now();
    stats.total_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
    stats.generated_tokens = n_past - tokens.size(); // approximate
    peak_memory_ = std::max(peak_memory_.load(), get_peak_memory());

    callback("", false, true, stats);
}

// ---------------------------------------------------------------------
// Peak memory (placeholder)
// ---------------------------------------------------------------------
size_t InferenceEngine::get_peak_memory() const {
    return 0;
}

} // namespace arm_ai
