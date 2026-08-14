#include "inference_engine.hpp"
#include <llama.h>
#include <stdexcept>
#include <vector>
#include <string>

namespace arm_ai {

bool InferenceEngine::load_model(const std::string& model_path, int n_ctx, int n_threads, int n_gpu_layers) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    // Note: use_mmap and use_mlock have been removed from llama_model_params

    model_.reset(llama_load_model_from_file(model_path.c_str(), model_params));
    if (!model_) {
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    // Note: logits_all has been removed from llama_context_params

    // Use modern initialization function instead of deprecated llama_new_context_with_model
    ctx_.reset(llama_init_from_model(model_.get(), ctx_params));
    if (!ctx_) {
        return false;
    }

    return true;
}

std::vector<int> InferenceEngine::tokenize(const std::string& text, bool add_special) {
    // Retrieve vocab handle from the model
    const llama_vocab* vocab = llama_model_get_vocab(model_.get());
    
    int n_tokens = text.length() + 3;
    std::vector<llama_token> tokens(n_tokens);
    
    int n = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, true);
    }
    tokens.resize(n);
    return tokens;
}

std::string InferenceEngine::detokenize(const std::vector<int>& tokens) {
    // Retrieve vocab handle from the context's model
    const llama_vocab* vocab = llama_model_get_vocab(llama_get_model(ctx_.get()));
    std::string result;
    for (auto token : tokens) {
        char piece[256];
        int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (n > 0) {
            result.append(piece, n);
        }
    }
    return result;
}

std::string InferenceEngine::apply_chat_template(const std::vector<std::pair<std::string, std::string>>& chat_messages) {
    const llama_vocab* vocab = llama_model_get_vocab(model_.get());
    
    std::vector<llama_chat_message> messages;
    for (const auto& msg : chat_messages) {
        messages.push_back({msg.first.c_str(), msg.second.c_str()});
    }

    std::vector<char> buf(1024);
    int n = llama_chat_apply_template(vocab, nullptr, messages.data(), messages.size(), true, buf.data(), buf.size());
    if (n > (int)buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(vocab, nullptr, messages.data(), messages.size(), true, buf.data(), buf.size());
    }
    if (n < 0) {
        return "";
    }
    return std::string(buf.data(), n);
}

} // namespace arm_ai
