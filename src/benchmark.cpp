#include "benchmark_helper.hpp"
#include "inference_engine.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace arm_ai;

struct BenchmarkConfig {
  std::string model_path;
  std::vector<std::string> prompts = {
      "What is the capital of France?",
      "Explain the theory of relativity in simple terms.",
      "Write a short story about a robot learning to paint."};
  int n_ctx = 4096;
  int n_threads = -1;
  int n_warmup_runs = 2;
  int n_benchmark_runs = 5;
  int max_tokens = 128;
  float temperature = 0.7f;
  float top_p = 0.9f;
  bool verbose = true;
};

struct BenchmarkResult {
  double ttft_ms;
  double tokens_per_second;
  double prompt_processing_tps;
  size_t prompt_tokens;
  size_t generated_tokens;
  double peak_memory_mb;
  double total_time_ms;
};

struct BenchmarkSummary {
  std::vector<BenchmarkResult> runs;

  double ttft_mean() const {
    return SafeStats::mean(extract(&BenchmarkResult::ttft_ms));
  }
  double ttft_p50() const {
    return SafeStats::percentile(extract(&BenchmarkResult::ttft_ms), 50.0);
  }
  double ttft_p95() const {
    return SafeStats::percentile(extract(&BenchmarkResult::ttft_ms), 95.0);
  }
  double ttft_p99() const {
    return SafeStats::percentile(extract(&BenchmarkResult::ttft_ms), 99.0);
  }

  double tps_mean() const {
    return SafeStats::mean(extract(&BenchmarkResult::tokens_per_second));
  }
  double tps_p50() const {
    return SafeStats::percentile(extract(&BenchmarkResult::tokens_per_second),
                                 50.0);
  }
  double tps_p95() const {
    return SafeStats::percentile(extract(&BenchmarkResult::tokens_per_second),
                                 95.0);
  }

  double memory_mean() const {
    return SafeStats::mean(extract(&BenchmarkResult::peak_memory_mb));
  }

private:
  std::vector<double> extract(double BenchmarkResult::*member) const {
    std::vector<double> v;
    v.reserve(runs.size());
    for (const auto &r : runs)
      v.push_back(r.*member);
    return v;
  }
};

class Benchmark {
public:
  explicit Benchmark(const BenchmarkConfig &cfg) : config_(cfg) {}

  BenchmarkSummary run() {
    BenchmarkSummary summary;
    std::cout << "\n== Arm AI Benchmark ==\n"
              << "Model: " << config_.model_path << '\n'
              << "Runs: " << config_.n_benchmark_runs << '\n';

    InferenceEngine engine;
    if (!engine.load_model(config_.model_path, config_.n_ctx, config_.n_threads,
                           0)) {
      std::cerr << "Failed to load model\n";
      return summary;
    }

    for (int i = 0; i < config_.n_warmup_runs; ++i) {
      run_single(engine, config_.prompts[0], true);
    }

    const size_t total = config_.n_benchmark_runs * config_.prompts.size();
    size_t count = 0;

    for (int run = 0; run < config_.n_benchmark_runs; ++run) {
      for (const auto &prompt : config_.prompts) {
        std::cout << "\rBenchmark " << ++count << '/' << total << "... "
                  << std::flush;
        summary.runs.push_back(run_single(engine, prompt, false));
      }
    }
    std::cout << "\n\n";
    return summary;
  }

private:
  BenchmarkResult run_single(InferenceEngine &engine, const std::string &prompt,
                             bool silent) {
    ChatRequest request;
    request.messages.push_back({"user", prompt});
    request.max_tokens = config_.max_tokens;
    request.temperature = config_.temperature;
    request.top_p = config_.top_p;
    request.stream = false;

    ChatResponse response = engine.chat_completion(request);

    BenchmarkResult result;
    result.ttft_ms =
        static_cast<double>(response.stats.time_to_first_token.count()) /
        1000.0;
    result.tokens_per_second = response.stats.tokens_per_second();
    result.prompt_processing_tps =
        response.stats.prompt_processing_tokens_per_second();
    result.prompt_tokens = response.stats.prompt_tokens;
    result.generated_tokens = response.stats.generated_tokens;
    result.peak_memory_mb =
        static_cast<double>(response.stats.peak_memory_bytes) /
        (1024.0 * 1024.0);
    result.total_time_ms =
        static_cast<double>(response.stats.total_inference_time.count()) /
        1000.0;

    return result;
  }
  BenchmarkConfig config_;
};

void print_summary(const BenchmarkSummary &summary) {
  std::cout << "== Benchmark Summary ==\n"
            << "Runs: " << summary.runs.size() << "\n\n"
            << std::fixed << std::setprecision(2)
            << "Time to First Token (TTFT):\n"
            << "  Mean: " << summary.ttft_mean() << " ms\n"
            << "  p95:  " << summary.ttft_p95() << " ms\n\n"
            << "Tokens per Second (Throughput):\n"
            << "  Mean: " << summary.tps_mean() << " tok/s\n"
            << "  p95:  " << summary.tps_p95() << " tok/s\n\n"
            << "Peak Memory:\n"
            << "  Mean: " << summary.memory_mean() << " MB\n\n";
}

int main(int argc, char *argv[]) {
  BenchmarkConfig config;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if ((arg == "-m" || arg == "--model") && i + 1 < argc)
        config.model_path = argv[++i];
      else if ((arg == "-r" || arg == "--runs") && i + 1 < argc)
        config.n_benchmark_runs = std::stoi(argv[++i]);
      else if ((arg == "-t" || arg == "--tokens") && i + 1 < argc)
        config.max_tokens = std::stoi(argv[++i]);
      else if (arg == "--threads" && i + 1 < argc)
        config.n_threads = std::stoi(argv[++i]);
    }
  } catch (const std::exception &) {
    std::cerr << "Error: Invalid numeric argument provided.\n";
    return 1;
  }

  if (config.model_path.empty()) {
    std::cerr << "Error: model path is required (-m)\n";
    return 1;
  }

  Benchmark benchmark(config);
  print_summary(benchmark.run());
  return 0;
}
