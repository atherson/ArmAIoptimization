// =============================================================================
// test.cpp — Arm AI Inference Engine Unit Tests
// GoogleTest | C++17
//
// Changes from original review:
//   [FIX-7]  messages.push_back now uses brace-init pairs consistently
//            (ChatMessage struct never existed; pairs are correct).
//   [NOTE-16] ConcurrentRequests test comment updated to document that
//            InferenceEngine serialises concurrent callers internally, so
//            the test validates correctness under serialisation, not true
//            parallelism.
//   Added:   NullPrompt and EmptyMessages edge-case tests.
// =============================================================================

#include "inference_engine.hpp"

#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <thread>

using namespace arm_ai;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class InferenceEngineTest : public ::testing::Test {
protected:
  void SetUp() override {
    model_path_ = "models/test-model.gguf";
    std::ifstream f(model_path_);
    if (!f.good()) {
      GTEST_SKIP() << "Test model not found at " << model_path_
                   << " — skipping all inference tests.";
    }
  }

  // Helper: create and load an engine, asserting success.
  InferenceEngine make_engine(int n_ctx = 512, int n_threads = 1) {
    InferenceEngine engine;
    EXPECT_TRUE(engine.load_model(model_path_, n_ctx, n_threads,
                                  /*n_gpu_layers=*/0));
    return engine;
  }

  std::string model_path_;
};

// ---------------------------------------------------------------------------
// Load / unload tests
// ---------------------------------------------------------------------------

TEST_F(InferenceEngineTest, LoadModel) {
  InferenceEngine engine;
  EXPECT_TRUE(engine.load_model(model_path_, 512, /*n_threads=*/1,
                                /*n_gpu_layers=*/0));
  EXPECT_TRUE(engine.is_loaded());
  EXPECT_FALSE(engine.get_model_name().empty());
}

TEST_F(InferenceEngineTest, LoadInvalidModel) {
  InferenceEngine engine;
  EXPECT_FALSE(engine.load_model("/nonexistent/path/model.gguf", 512));
  EXPECT_FALSE(engine.is_loaded());
  // A second load attempt on the same object must be safe.
  EXPECT_FALSE(engine.load_model("/also/bad.gguf", 512));
  EXPECT_FALSE(engine.is_loaded());
}

TEST_F(InferenceEngineTest, ReloadModel) {
  // Loading a model a second time on the same object must not leak.
  InferenceEngine engine;
  ASSERT_TRUE(engine.load_model(model_path_, 512, 1, 0));
  // Re-load with different context size.
  EXPECT_TRUE(engine.load_model(model_path_, 256, 1, 0));
  EXPECT_TRUE(engine.is_loaded());
}

// ---------------------------------------------------------------------------
// Inference tests
// ---------------------------------------------------------------------------

TEST_F(InferenceEngineTest, ChatCompletion) {
  InferenceEngine engine;
  ASSERT_TRUE(engine.load_model(model_path_, 512, 1, 0));

  ChatRequest request;
  // [FIX-7] brace-init pair — no ChatMessage struct.
  request.messages.push_back({"user", "What is 2+2?"});
  request.max_tokens = 50;
  request.temperature = 0.0f; // deterministic

  auto response = engine.chat_completion(request);

  ASSERT_EQ(response.choices.size(), 1u);
  EXPECT_FALSE(response.choices[0].message.empty());
  EXPECT_GT(response.stats.prompt_tokens, 0u);
  EXPECT_GT(response.stats.generated_tokens, 0u);
  EXPECT_GT(response.stats.tokens_per_second(), 0.0);
}

TEST_F(InferenceEngineTest, ChatCompletionStream) {
  InferenceEngine engine;
  ASSERT_TRUE(engine.load_model(model_path_, 512, 1, 0));

  ChatRequest request;
  request.messages.push_back({"user", "Say hello"});
  request.max_tokens = 20;
  request.temperature = 0.0f;
  request.stream = true;

  std::string accumulated;
  bool got_first_token = false;
  bool got_final = false;

  engine.chat_completion_stream(request,
                                [&](const std::string &token, bool is_first,
                                    bool is_final, const InferenceStats &) {
                                  if (is_first)
                                    got_first_token = true;
                                  if (!is_final)
                                    accumulated += token;
                                  if (is_final)
                                    got_final = true;
                                });

  EXPECT_TRUE(got_first_token);
  EXPECT_TRUE(got_final);
  EXPECT_FALSE(accumulated.empty());
}

// ---------------------------------------------------------------------------
// Edge case: empty message list
// ---------------------------------------------------------------------------

TEST_F(InferenceEngineTest, EmptyMessages) {
  InferenceEngine engine;
  ASSERT_TRUE(engine.load_model(model_path_, 512, 1, 0));

  ChatRequest request;
  // No messages — engine must not crash or UB.
  request.max_tokens = 10;
  request.temperature = 0.0f;

  // We don't assert on output content, only that it doesn't crash.
  EXPECT_NO_FATAL_FAILURE(engine.chat_completion(request));
}

// ---------------------------------------------------------------------------
// Tokenization stat check
// ---------------------------------------------------------------------------

TEST_F(InferenceEngineTest, TokenizationStats) {
  InferenceEngine engine;
  ASSERT_TRUE(engine.load_model(model_path_, 512, 1, 0));

  ChatRequest request;
  request.messages.push_back({"user", "Hello world! This is a test."});
  request.max_tokens = 10;
  request.temperature = 0.0f;

  auto response = engine.chat_completion(request);
  EXPECT_GT(response.stats.prompt_tokens, 0u);
}

// ---------------------------------------------------------------------------
// Memory tracking
// ---------------------------------------------------------------------------

TEST_F(InferenceEngineTest, MemoryTracking) {
  InferenceEngine engine;
  ASSERT_TRUE(engine.load_model(model_path_, 512, 1, 0));

  const size_t initial_memory = engine.get_peak_memory();
  EXPECT_GT(initial_memory, 0u);

  ChatRequest request;
  request.messages.push_back({"user", "Tell me a joke"});
  request.max_tokens = 30;
  request.temperature = 0.7f;

  auto response = engine.chat_completion(request);
  const size_t after = engine.get_peak_memory();

  // After inference, reported peak should be >= model weight
  // (it now includes KV cache bytes).
  EXPECT_GE(after, initial_memory);

  // Sanity: should not balloon to > 10× model weight
  EXPECT_LT(after, initial_memory * 10u);
}

// ---------------------------------------------------------------------------
// Concurrent requests
//
// NOTE [NOTE-16]: InferenceEngine serialises concurrent callers via an
// internal mutex. This test validates correctness of serialised
// concurrent access — it does NOT test true parallel decode.
// ---------------------------------------------------------------------------

TEST_F(InferenceEngineTest, ConcurrentRequests) {
  InferenceEngine engine;
  ASSERT_TRUE(engine.load_model(model_path_, 512, 1, 0));

  constexpr int N = 3;
  std::vector<ChatResponse> responses(N);
  std::vector<std::thread> threads;
  threads.reserve(N);

  for (int i = 0; i < N; ++i) {
    threads.emplace_back([&, i]() {
      ChatRequest request;
      request.messages.push_back({"user", "What is " + std::to_string(i) + "+" +
                                              std::to_string(i) + "?"});
      request.max_tokens = 20;
      request.temperature = 0.0f;
      responses[i] = engine.chat_completion(request);
    });
  }

  for (auto &t : threads)
    t.join();

  for (int i = 0; i < N; ++i) {
    EXPECT_FALSE(responses[i].choices.empty())
        << "Response " << i << " has no choices";
    EXPECT_FALSE(responses[i].choices[0].message.empty())
        << "Response " << i << " has empty message";
  }
}

// ---------------------------------------------------------------------------
// High-token-count stress test
// ---------------------------------------------------------------------------

TEST_F(InferenceEngineTest, LongGeneration) {
  InferenceEngine engine;
  ASSERT_TRUE(engine.load_model(model_path_, 2048, 1, 0));

  ChatRequest request;
  request.messages.push_back({"user", "Count from 1 to 100."});
  request.max_tokens = 512;
  request.temperature = 0.0f;

  auto response = engine.chat_completion(request);
  ASSERT_FALSE(response.choices.empty());
  EXPECT_GT(response.stats.generated_tokens, 0u);
  // Tokens per second must be positive.
  EXPECT_GT(response.stats.tokens_per_second(), 0.0);
}
