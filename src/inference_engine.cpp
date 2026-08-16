// =============================================================================
// inference_engine.cpp — Arm AI Inference Engine
// C++17 | llama.cpp backend | Armv9-A / KleidiAI optimised
//
// Bug fixes applied:
//   [FIX-1]  vocab_ null-guarded; partial-init state is safe.
//   [FIX-2]  Dead private helper declarations removed (see header).
//   [FIX-3]  peak_memory now tracks model weight + runtime KV cache bytes.
//   [FIX-4]  Sampler chain corrected: temp → top_p → dist (stochastic).
//            greedy used only when temperature == 0.0f.
//   [FIX-5]  llama_token_to_piece return value checked on both attempts;
//            no uninitialized bytes appended on repeated failure.
//   [FIX-6]  Sampler wrapped in SamplerPtr (RAII) — no leak on exceptions.
//   [FIX-10] engine_mutex_ removed from server (not here, but acknowledged).
//   [NOTE-8] Single-sequence (seq_id=0) documented.
// =============================================================================

#include "inference_engine.hpp"

#include <chrono>
#include <iostream>
#include <thread>

namespace arm_ai {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

InferenceEngine::InferenceEngine() = default;
InferenceEngine::~InferenceEngine() = default;

// ---------------------------------------------------------------------------
// load_model
// ---------------------------------------------------------------------------

bool InferenceEngine::load_model(const std::string& model_path,
                                 int n_ctx,
                                 int n_threads,
                                 int n_gpu_layers) {
    // Reset state so a second call to load_model is safe.
    ctx_.reset();
    model_.reset();
    vocab_ = nullptr;
    model_weight_bytes_.store(0, std::memory_order_relaxed);
    runtime_peak_bytes_.store(0,  std::memory_order_relaxed);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = n_gpu_layers;

    llama_model* raw_model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!raw_model) {
        std::cerr << "[InferenceEngine] Failed to load model from: " << model_path << '\n';
        return false;
    }
    model_.reset(raw_model);

    // [FIX-1] vocab_ lifetime is tied to model_; only assigned after model
    // is successfully wrapped in the unique_ptr.
    vocab_ = llama_model_get_vocab(model_.get());

    const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    const int threads    = (n_threads > 0) ? n_threads : hw_threads;

    llama_context_params ctx_params  = llama_context_default_params();
    ctx_params.n_ctx                  = n_ctx;
    ctx_params.n_batch                = 512;
    ctx_params.n_threads              = threads;
    ctx_params.n_threads_batch        = threads;

    llama_context* raw_ctx = llama_init_from_model(model_.get(), ctx_params);
    if (!raw_ctx) {
        std::cerr << "[InferenceEngine] Failed to create llama context\n";
        // model_ will be freed by unique_ptr; vocab_ becomes a dangling
        // pointer only if we leave it set — clear it. [FIX-1]
        vocab_ = nullptr;
        model_.reset();
        return false;
    }
    ctx_.reset(raw_ctx);

    model_name_   = model_path;
    n_ctx_        = n_ctx;
    n_threads_    = threads;

    // [FIX-3] Record model weight bytes at load time.
    model_weight_bytes_.store(llama_model_size(model_.get()), std::memory_order_relaxed);
    return true;
}

// ---------------------------------------------------------------------------
// get_peak_memory — returns model weight + last observed runtime KV bytes.
// [FIX-3]
// ---------------------------------------------------------------------------

size_t InferenceEngine::get_peak_memory() const {
    if (!model_) return 0;
    return model_weight_bytes_.load(std::memory_order_relaxed)
         + runtime_peak_bytes_.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

// [FIX-4] Build an appropriate sampler chain.
// For temperature == 0 we use greedy (deterministic, fastest).
// For temperature > 0 we use: temperature → top_p → distribution sampler
// so that both parameters actually influence the output.
SamplerPtr InferenceEngine::make_sampler(float temperature, float top_p) const {
    SamplerPtr chain(
        llama_sampler_chain_init(llama_sampler_chain_default_params()));

    if (temperature == 0.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain.get(),
            llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(chain.get(),
            llama_sampler_init_top_p(top_p, /*min_keep=*/1));
        // dist sampler draws from the reshaped distribution. [FIX-4]
        llama_sampler_chain_add(chain.get(),
            llama_sampler_init_dist(/*seed=*/LLAMA_DEFAULT_SEED));
    }
    return chain;
}

// [FIX-5] Tokenize a prompt string into out_tokens.
// Returns false if llama_tokenize fails twice.
bool InferenceEngine::tokenize_prompt(const std::string& prompt,
                                      std::vector<llama_token>& out_tokens) const {
    out_tokens.resize(prompt.size() + 64); // +64 for BOS / special tokens
    int n = llama_tokenize(model_.get(),
                           prompt.c_str(),
                           static_cast<int>(prompt.size()),
                           out_tokens.data(),
                           static_cast<int>(out_tokens.size()),
                           /*add_special=*/true,
                           /*parse_special=*/true);
    if (n < 0) {
        // Buffer was too small — resize to the required length and retry.
        out_tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(model_.get(),
                           prompt.c_str(),
                           static_cast<int>(prompt.size()),
                           out_tokens.data(),
                           static_cast<int>(out_tokens.size()),
                           true, true);
        if (n < 0) {
            std::cerr << "[InferenceEngine] tokenize failed (required "
                      << -n << " slots)\n";
            return false;
        }
    }
    out_tokens.resize(static_cast<size_t>(n));
    return true;
}

// [FIX-5] Convert a single token id to its UTF-8 string piece.
// Returns an empty string on failure instead of appending garbage bytes.
std::string InferenceEngine::token_to_piece(llama_token id) const {
    char buf[256];
    int n = llama_token_to_piece(model_.get(), id, buf, sizeof(buf), 0, true);
    if (n > 0) {
        return std::string(buf, static_cast<size_t>(n));
    }
    if (n < 0) {
        // Token piece is larger than 256 bytes — rare but possible.
        std::vector<char> large_buf(static_cast<size_t>(-n));
        int n2 = llama_token_to_piece(model_.get(), id,
                                      large_buf.data(),
                                      large_buf.size(), 0, true);
        if (n2 > 0) {
            return std::string(large_buf.data(), static_cast<size_t>(n2));
        }
        // Still failed — return empty so caller can skip gracefully.
        std::cerr << "[InferenceEngine] token_to_piece failed for id " << id << '\n';
    }
    return {};
}

// ---------------------------------------------------------------------------
// chat_completion — blocking inference
// ---------------------------------------------------------------------------

ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);

    ChatResponse response;

    if (!is_loaded()) {
        std::cerr << "[InferenceEngine] chat_completion called but model not loaded\n";
        return response;
    }

    llama_kv_cache_clear(ctx_.get());

    // Build prompt string from the message list.
    std::string prompt;
    prompt.reserve(512);
    for (const auto& [role, content] : request.messages) {
        prompt += role;
        prompt += ": ";
        prompt += content;
        prompt += '\n';
    }
    prompt += "assistant: ";

    // Tokenise. [FIX-5]
    std::vector<llama_token> tokens;
    if (!tokenize_prompt(prompt, tokens)) {
        return response;
    }

    InferenceStats stats;
    stats.prompt_tokens = tokens.size();

    // --- Prompt processing ---
    const auto t_prompt_start = std::chrono::steady_clock::now();

    llama_batch batch = llama_batch_get_one(tokens.data(),
                                            static_cast<int>(tokens.size()));
    if (llama_decode(ctx_.get(), batch) != 0) {
        std::cerr << "[InferenceEngine] Failed to decode prompt\n";
        return response;
    }

    const auto t_prompt_end = std::chrono::steady_clock::now();
    stats.prompt_processing_time =
        std::chrono::duration_cast<std::chrono::microseconds>(
            t_prompt_end - t_prompt_start);

    // --- Sampler (RAII — no leak on any exit path). [FIX-4] [FIX-6] ---
    SamplerPtr smpl = make_sampler(request.temperature, request.top_p);

    // --- Generation loop ---
    std::string generated_text;
    generated_text.reserve(static_cast<size_t>(request.max_tokens) * 4);

    const auto t_gen_start = std::chrono::steady_clock::now();
    bool first_token = true;

    for (int i = 0; i < request.max_tokens; ++i) {
        llama_token id = llama_sampler_sample(smpl.get(), ctx_.get(), -1);
        llama_sampler_accept(smpl.get(), id);

        if (llama_token_is_eog(model_.get(), id)) break;

        if (first_token) {
            stats.time_to_first_token =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_prompt_end);
            first_token = false;
        }

        // [FIX-5] token_to_piece returns "" on failure; nothing appended.
        generated_text += token_to_piece(id);
        stats.generated_tokens++;

        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_.get(), batch) != 0) {
            std::cerr << "[InferenceEngine] decode failed at token " << i << '\n';
            break;
        }
    }

    const auto t_gen_end = std::chrono::steady_clock::now();
    stats.total_inference_time =
        std::chrono::duration_cast<std::chrono::microseconds>(
            t_gen_end - t_gen_start);

    // [FIX-3] Update runtime peak with KV cache state size after generation.
    const size_t kv_size = llama_state_get_size(ctx_.get());
    runtime_peak_bytes_.store(kv_size, std::memory_order_relaxed);
    stats.peak_memory_bytes = get_peak_memory();

    // smpl freed here by SamplerPtr destructor. [FIX-6]

    ChatChoice choice;
    choice.index        = 0;
    choice.message      = std::move(generated_text);
    choice.finish_reason = "stop";
    response.choices.push_back(std::move(choice));
    response.stats = stats;

    return response;
}

// ---------------------------------------------------------------------------
// chat_completion_stream — streaming inference
// ---------------------------------------------------------------------------

void InferenceEngine::chat_completion_stream(
    const ChatRequest& request,
    std::function<void(const std::string&, bool, bool,
                       const InferenceStats&)> callback) {

    // NOTE: The lock is held for the entire stream duration. If callback
    // performs slow network I/O, the engine is blocked for all other
    // callers. A future improvement is to batch tokens, release the lock,
    // flush, and re-acquire. [NOTE-11]
    std::lock_guard<std::mutex> lock(mutex_);

    InferenceStats stats;

    if (!is_loaded()) {
        std::cerr << "[InferenceEngine] chat_completion_stream called but model not loaded\n";
        callback("", false, true, stats);
        return;
    }

    llama_kv_cache_clear(ctx_.get());

    std::string prompt;
    prompt.reserve(512);
    for (const auto& [role, content] : request.messages) {
        prompt += role;
        prompt += ": ";
        prompt += content;
        prompt += '\n';
    }
    prompt += "assistant: ";

    std::vector<llama_token> tokens;
    if (!tokenize_prompt(prompt, tokens)) {
        callback("", false, true, stats);
        return;
    }

    stats.prompt_tokens = tokens.size();

    // Prompt processing
    const auto t_prompt_start = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(tokens.data(),
                                            static_cast<int>(tokens.size()));
    if (llama_decode(ctx_.get(), batch) != 0) {
        std::cerr << "[InferenceEngine] Failed to decode prompt (stream)\n";
        callback("", false, true, stats);
        return;
    }
    const auto t_prompt_end = std::chrono::steady_clock::now();
    stats.prompt_processing_time =
        std::chrono::duration_cast<std::chrono::microseconds>(
            t_prompt_end - t_prompt_start);

    // [FIX-4] [FIX-6] RAII sampler
    SamplerPtr smpl = make_sampler(request.temperature, request.top_p);

    const auto t_gen_start = std::chrono::steady_clock::now();
    bool is_first = true;

    for (int i = 0; i < request.max_tokens; ++i) {
        llama_token id = llama_sampler_sample(smpl.get(), ctx_.get(), -1);
        llama_sampler_accept(smpl.get(), id);

        if (llama_token_is_eog(model_.get(), id)) {
            stats.total_inference_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_gen_start);
            // [FIX-3]
            runtime_peak_bytes_.store(llama_state_get_size(ctx_.get()),
                                      std::memory_order_relaxed);
            stats.peak_memory_bytes = get_peak_memory();
            callback("", false, true, stats);
            return;
        }

        if (is_first) {
            stats.time_to_first_token =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_prompt_end);
        }

        // [FIX-5] Safe token piece conversion
        const std::string piece = token_to_piece(id);
        stats.generated_tokens++;

        callback(piece, is_first, false, stats);
        is_first = false;

        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_.get(), batch) != 0) {
            std::cerr << "[InferenceEngine] decode failed at token " << i
                      << " (stream)\n";
            stats.total_inference_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_gen_start);
            runtime_peak_bytes_.store(llama_state_get_size(ctx_.get()),
                                      std::memory_order_relaxed);
            stats.peak_memory_bytes = get_peak_memory();
            callback("", false, true, stats);
            return;
        }
    }

    // max_tokens reached without EOS
    stats.total_inference_time =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_gen_start);
    runtime_peak_bytes_.store(llama_state_get_size(ctx_.get()),
                              std::memory_order_relaxed);
    stats.peak_memory_bytes = get_peak_memory();
    callback("", false, true, stats);

    // smpl freed here by SamplerPtr destructor. [FIX-6]
}

} // namespace arm_ai
