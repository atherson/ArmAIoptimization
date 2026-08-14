#include "inference_engine.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace arm_ai {

InferenceEngine::InferenceEngine() {}

InferenceEngine::~InferenceEngine() {
    // unique_ptr deleters (llama_free / llama_model_free) handle cleanup automatically.
}

// -----------------------------------------------------------------------
// BUG 1 FIXED: Parameter list was (model_path, context_size, batch_size,
//   n_threads) but the header declares (model_path, n_ctx, n_threads,
//   n_gpu_layers).  The implementation was mapping batch_size→n_threads and
//   n_threads→n_gpu_layers, so threads were always 0 and n_gpu_layers was
//   ignored.  Corrected parameter names + order to match the header.
//
// BUG 2 FIXED: vocab_ was never assigned after model load, making every
//   future call that would use it (tokenise, detokenise, chat-template)
//   operate on a null pointer.  Set it here via llama_model_get_vocab().
// -----------------------------------------------------------------------
bool InferenceEngine::load_model(const std::string& model_path,
                                  int n_ctx,
                                  int n_threads,
                                  int n_gpu_layers)
{
    model_name_ = model_path;
    n_ctx_      = n_ctx;
    n_threads_  = n_threads;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = n_gpu_layers;          // BUG 1 fix

    struct llama_model* model_ptr =
        llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model_ptr) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return false;
    }
    model_.reset(model_ptr);

    vocab_ = llama_model_get_vocab(model_.get());            // BUG 2 fix

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx            = n_ctx;
    ctx_params.n_batch          = 512;                       // sensible default
    ctx_params.n_threads        = n_threads > 0
                                      ? n_threads
                                      : static_cast<int>(std::thread::hardware_concurrency());
    ctx_params.n_threads_batch  = ctx_params.n_threads;

    struct llama_context* ctx_ptr =
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
    if (!model_) return 0;
    return llama_model_size(model_.get());
}

// -----------------------------------------------------------------------
// BUG 3 FIXED: chat_completion() wrote its result to `response.content`
//   which does not exist on ChatResponse.  The struct has `choices`
//   (vector<ChatChoice>) and `stats`.  Push a ChatChoice and assign stats.
//
// BUG 4 FIXED: InferenceStats timing was never recorded, so
//   tokens_per_second() always returned 0 and benchmark TTFT was always 0.
//   Added wall-clock measurements for prompt-processing and generation.
//
// BUG 5 FIXED: is_first flag in stream callback was always true after the
//   first token — the boolean was never toggled.  Fixed with a local flag.
// -----------------------------------------------------------------------
ChatResponse InferenceEngine::chat_completion(const ChatRequest& request) {
    ChatResponse response;

    if (!is_loaded()) {
        std::cerr << "Model not loaded" << std::endl;
        return response;
    }

    llama_kv_cache_clear(ctx_.get());

    // Build a simple role: content prompt (no external template dependency)
    std::string prompt;
    for (const auto& msg : request.messages) {
        prompt += msg.first + ": " + msg.second + "\n";
    }
    prompt += "assistant: ";

    // Tokenise
    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(model_.get(),
                                   prompt.c_str(), static_cast<int>(prompt.size()),
                                   tokens.data(),  static_cast<int>(tokens.size()),
                                   /*add_special=*/true, /*parse_special=*/true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(model_.get(),
                                   prompt.c_str(), static_cast<int>(prompt.size()),
                                   tokens.data(),  static_cast<int>(tokens.size()),
                                   true, true);
    }
    tokens.resize(n_tokens);

    InferenceStats stats;
    stats.prompt_tokens = static_cast<size_t>(n_tokens);

    // --- Prompt processing ---
    auto t_prompt_start = std::chrono::steady_clock::now();

    llama_batch batch = llama_batch_get_one(tokens.data(),
                                             static_cast<int>(tokens.size()));
    if (llama_decode(ctx_.get(), batch) != 0) {
        std::cerr << "Failed to decode prompt" << std::endl;
        return response;
    }

    auto t_prompt_end = std::chrono::steady_clock::now();
    stats.prompt_processing_time =
        std::chrono::duration_cast<std::chrono::microseconds>(
            t_prompt_end - t_prompt_start);

    // --- Generation ---
    struct llama_sampler* smpl =
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

        if (llama_token_is_eog(model_.get(), id)) break;

        if (first_token) {
            stats.time_to_first_token =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_prompt_end);  // BUG 4 fix
            first_token = false;
        }

        char buf[256];
        int n = llama_token_to_piece(model_.get(), id, buf, sizeof(buf), 0, true);
        if (n > 0) generated_text.append(buf, n);

        stats.generated_tokens++;

        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_.get(), batch) != 0) break;
    }

    auto t_gen_end = std::chrono::steady_clock::now();
    stats.total_inference_time =
        std::chrono::duration_cast<std::chrono::microseconds>(
            t_gen_end - t_gen_start);
    stats.peak_memory_bytes = get_peak_memory();

    llama_sampler_free(smpl);

    // BUG 3 fix: populate choices, not a non-existent .content field
    ChatChoice choice;
    choice.index        = 0;
    choice.message      = generated_text;
    choice.finish_reason = "stop";
    response.choices.push_back(choice);
    response.stats = stats;                                  // BUG 3 fix

    return response;
}

void InferenceEngine::chat_completion_stream(
    const ChatRequest& request,
    std::function<void(const std::string&, bool, bool, const InferenceStats&)> callback)
{
    if (!is_loaded()) {
        InferenceStats empty;
        callback("", false, true, empty);
        return;
    }

    llama_kv_cache_clear(ctx_.get());

    std::string prompt;
    for (const auto& msg : request.messages) {
        prompt += msg.first + ": " + msg.second + "\n";
    }
    prompt += "assistant: ";

    std::vector<llama_token> tokens(prompt.size() + 32);
    int n_tokens = llama_tokenize(model_.get(),
                                   prompt.c_str(), static_cast<int>(prompt.size()),
                                   tokens.data(),  static_cast<int>(tokens.size()),
                                   true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(model_.get(),
                                   prompt.c_str(), static_cast<int>(prompt.size()),
                                   tokens.data(),  static_cast<int>(tokens.size()),
                                   true, true);
    }
    tokens.resize(n_tokens);

    InferenceStats stats;
    stats.prompt_tokens = static_cast<size_t>(n_tokens);

    auto t_prompt_end = std::chrono::steady_clock::now();   // will be set after decode

    llama_batch batch = llama_batch_get_one(tokens.data(),
                                             static_cast<int>(tokens.size()));
    if (llama_decode(ctx_.get(), batch) != 0) {
        callback("", false, true, stats);
        return;
    }
    t_prompt_end = std::chrono::steady_clock::now();

    struct llama_sampler* smpl =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(request.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(request.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    auto t_gen_start = std::chrono::steady_clock::now();
    bool is_first    = true;           // BUG 5 fix: track first-token flag properly

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
        int  n = llama_token_to_piece(model_.get(), id, buf, sizeof(buf), 0, true);
        std::string piece = (n > 0) ? std::string(buf, n) : "";

        stats.generated_tokens++;
        callback(piece, is_first, false, stats);    // BUG 5 fix: pass real is_first

        is_first = false;                           // only true for the very first token

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
