#include "inference_engine.hpp"
#include <llama.h>
#include <iostream>

namespace arm_ai {

bool InferenceEngine::load_model(const std::string& model_path, int n_ctx, int n_gpu_layers, int n_threads) {
    if (model_) return true;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    model_.reset(llama_model_load_from_file(model_path.c_str(), model_params));  // <-- updated
    if (!model_) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return false;
    }

    // ... rest of loading, context creation etc.
    return true;
}

std::string InferenceEngine::apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages) {
    if (!model_) return "";

    std::vector<llama_chat_message> msgs;
    msgs.reserve(messages.size());
    for (const auto& [role, content] : messages) {
        msgs.push_back({role.c_str(), content.c_str()});
    }

    // Get required buffer size
    int n = llama_chat_apply_template(nullptr, msgs.data(), msgs.size(), true, nullptr, 0);
    if (n < 0) return "";

    std::string buf(n + 1, '\0');
    n = llama_chat_apply_template(nullptr, msgs.data(), msgs.size(), true, buf.data(), buf.size());
    if (n < 0) return "";
    buf.resize(n);
    return buf;
}

} // namespace arm_ai
