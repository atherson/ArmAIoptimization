// =============================================================================
// benchmark.cpp — Arm AI Inference Benchmark
// C++17 | Measures TTFT, throughput, and memory across N runs.
//
// Changes from original review:
//   [FIX-14] n_gpu_layers=0 now explicit (was relying on default) for
//            clarity on CPU-only Arm targets.
//   Minor: Sampler / peak-memory changes are transparent — they land in
//   inference_engine.cpp, not here. Benchmark output remains identical.
// =============================================================================

#include "inference_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

using namespace arm_ai;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct BenchmarkConfig {
  std::string model_path;
  std::vector<std::string> prompts = {
      "What is the capital of France?",
      "Explain the theory of relativity in simple terms.",
      "Write a short story about a robot learning to paint.",
      "What are the benefits of renewable energy?",
      "Describe the process of photosynthesis."};
  int n_ctx = 4096;
  int n_threads = -1;
  int n_warmup_runs = 2;
  int n_benchmark_runs = 5;
  int max_tokens = 128;
  float temperature = 0.7f;
  float top_p = 0.9f;
  bool verbose = true;
};

// ---------------------------------------------------------------------------
// Per-run result
// ---------------------------------------------------------------------------

struct BenchmarkResult {
  double ttft_ms;
  double tokens_per_second;
  double prompt_processing_tps;
  size_t prompt_tokens;
  size_t generated_tokens;
  double peak_memory_mb; // FP to avoid truncation (was size_t in original)
  double total_time_ms;
};

// ---------------------------------------------------------------------------
// Summary statistics over all runs
// ---------------------------------------------------------------------------

struct BenchmarkSummary {
  std::vector<BenchmarkResult> runs;

  double ttft_mean() const { return mean(ttft_ms_vec()); }
  double ttft_p50() const { return percentile(ttft_ms_vec(), 50.0); }
  double ttft_p95() const { return percentile(ttft_ms_vec(), 95.0); }
  double ttft_p99() const { return percentile(ttft_ms_vec(), 99.0); }

  double tps_mean() const { return mean(tps_vec()); }
  double tps_p50() const { return percentile(tps_vec(), 50.0); }
  double tps_p95() const { return percentile(tps_vec(), 95.0); }

  double memory_mean() const { return mean(mem_vec()); }

private:
  std::vector<double> ttft_ms_vec() const {
    std::vector<double> v;
    v.reserve(runs.size());
    for (const auto &r : runs)
      v.push_back(r.ttft_ms);
    return v;
  }
  std::vector<double> tps_vec() const {
    std::vector<double> v;
    v.reserve(runs.size());
    for (const auto &r : runs)
      v.push_back(r.tokens_per_second);
    return v;
  }
  std::vector<double> mem_vec() const {
    std::vector<double> v;
    v.reserve(runs.size());
    for (const auto &r : runs)
      v.push_back(r.peak_memory_mb);
    return v;
  }

  static double mean(const std::vector<double> &v) {
    if (v.empty())
      return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) /
           static_cast<double>(v.size());
  }

  static double percentile(std::vector<double> v, double p) {
    if (v.empty())
      return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx =
        std::min(static_cast<size_t>(
                     std::ceil(p / 100.0 * static_cast<double>(v.size()))) -
                     1u,
                 v.size() - 1u);
    return v[idx];
  }
};

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

class Benchmark {
public:
  explicit Benchmark(const BenchmarkConfig &cfg) : config_(cfg) {}

  BenchmarkSummary run() {
    BenchmarkSummary summary;

    std::cout << "\n== Arm AI Benchmark ==\n"
              << "Model:      " << config_.model_path << '\n'
              << "Runs:       " << config_.n_benchmark_runs
              << " (warmup: " << config_.n_warmup_runs << ")\n"
              << "Prompts:    " << config_.prompts.size() << '\n'
              << "Max tokens: " << config_.max_tokens << "\n\n";

    InferenceEngine engine;

    // [FIX-14] Explicit n_gpu_layers=0 for CPU-only Arm targets.
    if (!engine.load_model(config_.model_path, config_.n_ctx, config_.n_threads,
                           /*n_gpu_layers=*/0)) {
      std::cerr << "[Benchmark] Failed to load model: " << config_.model_path
                << '\n';
      return summary;
    }

    std::cout << "Model loaded: " << engine.get_model_name() << "\n\n";

    // Warmup runs — results discarded.
    for (int i = 0; i < config_.n_warmup_runs; ++i) {
      std::cout << "Warmup " << (i + 1) << '/' << config_.n_warmup_runs
                << "... " << std::flush;
      run_single(engine, config_.prompts[0], /*silent=*/true);
      std::cout << "done\n";
    }
    std::cout << '\n';

    const size_t total_runs =
        static_cast<size_t>(config_.n_benchmark_runs) * config_.prompts.size();
    size_t run_count = 0;

    for (int run = 0; run < config_.n_benchmark_runs; ++run) {
      for (const auto &prompt : config_.prompts) {
        ++run_count;
        std::cout << "\rBenchmark " << run_count << '/' << total_runs << "... "
                  << std::flush;
        summary.runs.push_back(run_single(engine, prompt, /*silent=*/false));
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

    if (!silent && config_.verbose) {
      const std::string prompt_preview =
          prompt.size() > 40 ? prompt.substr(0, 40) + "..." : prompt;

      // Guard against empty choices (e.g. model load but decode failed).
      const std::string response_preview =
          response.choices.empty()
              ? "(no response)"
              : (response.choices[0].message.size() > 80
                     ? response.choices[0].message.substr(0, 80) + "..."
                     : response.choices[0].message);

      std::cout << "\n  Prompt:          " << prompt_preview << '\n'
                << "  Response:        " << response_preview << '\n'
                << std::fixed << std::setprecision(2)
                << "  TTFT:            " << result.ttft_ms << " ms\n"
                << "  Tokens/sec:      " << result.tokens_per_second << "\n"
                << "  Prompt tokens:   " << result.prompt_tokens << '\n'
                << "  Gen tokens:      " << result.generated_tokens << '\n'
                << "  Peak memory:     " << result.peak_memory_mb << " MB\n\n";
    }

    return result;
  }

  BenchmarkConfig config_;
};

// ---------------------------------------------------------------------------
// Summary printer
// ---------------------------------------------------------------------------

void print_summary(const BenchmarkSummary &summary) {
  std::cout << "== Benchmark Summary ==\n"
            << "Runs: " << summary.runs.size() << "\n\n"
            << std::fixed << std::setprecision(2)
            << "Time to First Token (TTFT):\n"
            << "  Mean: " << summary.ttft_mean() << " ms\n"
            << "  p50:  " << summary.ttft_p50() << " ms\n"
            << "  p95:  " << summary.ttft_p95() << " ms\n"
            << "  p99:  " << summary.ttft_p99() << " ms\n\n"
            << "Tokens per Second (Throughput):\n"
            << "  Mean: " << summary.tps_mean() << " tok/s\n"
            << "  p50:  " << summary.tps_p50() << " tok/s\n"
            << "  p95:  " << summary.tps_p95() << " tok/s\n\n"
            << "Peak Memory:\n"
            << "  Mean: " << summary.memory_mean() << " MB\n\n";
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  BenchmarkConfig config;
  config.model_path = "models/model.gguf";

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
    else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: arm-ai-benchmark [options]\n"
                << "  -m, --model <path>    Model path (required)\n"
                << "  -r, --runs <num>      Benchmark runs (default: 5)\n"
                << "  -t, --tokens <num>    Max tokens per run (default: 128)\n"
                << "      --threads <num>   CPU threads (default: auto)\n"
                << "  -h, --help            Show this help\n";
      return 0;
    }
  }

  if (config.model_path.empty()) {
    std::cerr << "Error: model path is required (-m)\n";
    return 1;
  }

  Benchmark benchmark(config);
  const auto summary = benchmark.run();
  print_summary(summary);
  return 0;
}
