#include <gtest/gtest.h>   

#include "inference_engine.hpp"

#include <fstream>
#include <thread>
#include <chrono>

using namespace arm_ai;

// Test fixture
class InferenceEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_path_ = "models/test-model.gguf";
        std::ifstream f(model_path_);
        if (!f.good()) {
            GTEST_SKIP() << "Test model not found at " << model_path_;
        }
    }

    std::string model_path_;
};

TEST_F(InferenceEngineTest, LoadModel) {
    InferenceEngine engine;
    EXPECT_TRUE(engine.load_model(model_path_, 512, 1));
    EXPECT_TRUE(engine.is_loaded());
    EXPECT_FALSE(engine.get_model_name().empty());
}

TEST_F(InferenceEngineTest, LoadInvalidModel) {
    InferenceEngine engine;
    EXPECT_FALSE(engine.load_model("/nonexistent/model.gguf", 512));
    EXPECT_FALSE(engine.is_loaded());
}

TEST_F(InferenceEngineTest, ChatCompletion) {
    InferenceEngine engine;
    ASSERT_TRUE(engine.load_model(model_path_, 512, 1));

    ChatRequest request;
    request.messages.push_back({"user", "What is 2+2?"});
    request.max_tokens = 50;
    request.temperature = 0.0f;

    auto response = engine.chat_completion(request);

    EXPECT_EQ(response.choices.size(), 1);
    EXPECT_FALSE(response.choices[0].message.empty());
    EXPECT_GT(response.stats.prompt_tokens, 0);
    EXPECT_GT(response.stats.generated_tokens, 0);
    EXPECT_GT(response.stats.tokens_per_second(), 0);
}

TEST_F(InferenceEngineTest, ChatCompletionStream) {
    InferenceEngine engine;
    ASSERT_TRUE(engine.load_model(model_path_, 512, 1));

    ChatRequest request;
    request.messages.push_back({"user", "Say hello"});
    request.max_tokens = 20;
    request.temperature = 0.0f;
    request.stream = true;

    std::string accumulated;
    bool got_first_token = false;
    bool got_final = false;

    engine.chat_completion_stream(
        request,
        [&](const std::string& token, bool is_first, bool is_final, const InferenceStats&) {
            if (is_first) got_first_token = true;
            if (!is_final) accumulated += token;
            if (is_final) got_final = true;
        }
    );

    EXPECT_TRUE(got_first_token);
    EXPECT_TRUE(got_final);
    EXPECT_FALSE(accumulated.empty());
}

TEST_F(InferenceEngineTest, Tokenization) {
    InferenceEngine engine;
    ASSERT_TRUE(engine.load_model(model_path_, 512, 1));

    ChatRequest request;
    request.messages.push_back({"user", "Hello world! This is a test."});
    request.max_tokens = 10;
    request.temperature = 0.0f;

    auto response = engine.chat_completion(request);
    EXPECT_GT(response.stats.prompt_tokens, 0);
}

TEST_F(InferenceEngineTest, MemoryTracking) {
    InferenceEngine engine;
    ASSERT_TRUE(engine.load_model(model_path_, 512, 1));

    size_t initial_memory = engine.get_peak_memory();
    EXPECT_GT(initial_memory, 0);

    ChatRequest request;
    request.messages.push_back({"user", "Tell me a joke"});
    request.max_tokens = 30;
    request.temperature = 0.7f;

    auto response = engine.chat_completion(request);
    size_t after_memory = engine.get_peak_memory();

    EXPECT_LT(after_memory, initial_memory * 3);
}

TEST_F(InferenceEngineTest, ConcurrentRequests) {
    InferenceEngine engine;
    ASSERT_TRUE(engine.load_model(model_path_, 512, 1));

    auto run_request = [&](int id) {
        ChatRequest request;
        request.messages.push_back({"user", "What is " + std::to_string(id) + "+" + std::to_string(id) + "?"});
        request.max_tokens = 20;
        request.temperature = 0.0f;
        return engine.chat_completion(request);
    };

    std::vector<std::thread> threads;
    std::vector<ChatResponse> responses(3);

    for (int i = 0; i < 3; i++) {
        threads.emplace_back([&, i]() {
            responses[i] = run_request(i);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (const auto& resp : responses) {
        EXPECT_FALSE(resp.choices.empty());
        EXPECT_FALSE(resp.choices[0].message.empty());
    }
}
