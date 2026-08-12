#include "inference_engine.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cmath>

using namespace arm_ai;

struct BenchmarkConfig {
    std::string model_path;
    std::vector<std::string> prompts = {
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
    };
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
   size_t peak_memory_mb;
    double total_time_ms;
};

struct BenchmarkSummary {
    std::vector<BenchmarkResult> runs;
    
    double ttft_mean() const { return mean(ttft_ms()); }
    double ttft_p50() const { return percentile(ttft_ms(), 50); }
    double ttft_p95() const { return percentile(ttft_ms(), 95); }
    double ttft_p99() const { return percentile(ttft_ms(), 99); }
    
    double tps_mean() const { return mean(tokens_per_second()); }
    double tps_p50() const { return percentile(tokens_per_second(), 50); }
    double tps_p95() const { return percentile(tokens_per_second(), 95); }
    
    double memory_mean() const { return mean(peak_memory_mb()); }
    
    std::vector<double> ttft_ms() const {
        std::vector<double> v;
        v.reserve(runs.size());
        for (const auto& r : runs) v.push_back(r.ttft_ms);
        return v;
    }
    
    std::vector<double> tokens_per_second() const {
        std::vector<double> v;
        v.reserve(runs.size());
        for (const auto& r : runs) v.push_back(r.tokens_per_second);
        return v;
    }
    
    std::vector<double> peak_memory_mb() const {
        std::vector<double> v;
        v.reserve(runs.size());
        for (const auto& r : runs) v.push_back(r.peak_memory_mb);
        return v;
    }
    
    static double mean(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }
    
    static double percentile(std::vector<double> v, double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * v.size())) - 1;
        idx = std::min(idx, v.size() - 1);
        return v[idx];
    }
};

class Benchmark {
public:
    explicit Benchmark(const BenchmarkConfig& config) : config_(config) {}
    
    BenchmarkSummary run() {
        BenchmarkSummary summary;
        
        std::cout << "\n== Arm AI Benchmark ==" << std::endl;
        std::cout << "Model: " << config_.model_path << std::endl;
        std::cout << "Runs: " << config_.n_benchmark_runs << " (warmup: " 
                  << config_.n_warmup_runs << ")" << std::endl;
        std::cout << "Prompts: " << config_.prompts.size() << std::endl;
        std::cout << "Max tokens: " << config_.max_tokens << std::endl;
        std::cout << std::endl;
        
        InferenceEngine engine;
        
        if (!engine.load_model(config_.model_path, config_.n_ctx, config_.n_threads)) {
            std::cerr << "Failed to load model: " << config_.model_path << std::endl;
            return summary;
        }
        
        std::cout << "Model loaded: " << engine.get_model_name() << std::endl;
        std::cout << std::endl;
        
        for (int i = 0; i < config_.n_warmup_runs; i++) {
            std::cout << "Warmup " << (i + 1) << "/" << config_.n_warmup_runs << "... " << std::flush;
            run_single(engine, config_.prompts[0], true);
            std::cout << "done" << std::endl;
        }
        std::cout << std::endl;
        
        size_t total_runs = config_.n_benchmark_runs * config_.prompts.size();
        size_t run_count = 0;
        
        for (int run = 0; run < config_.n_benchmark_runs; run++) {
            for (const auto& prompt : config_.prompts) {
                run_count++;
                std::cout << "\rBenchmark " << run_count << "/" << total_runs << "... " << std::flush;
                
                auto result = run_single(engine, prompt, false);
                summary.runs.push_back(result);
            }
        }
        std::cout << "\n" << std::endl;
        
        return summary;
    }
    
private:
    BenchmarkResult run_single(InferenceEngine& engine, const std::string& prompt, bool silent) {
        ChatRequest request;
        request.messages.push_back({"user", prompt});
        request.max_tokens = config_.max_tokens;
        request.temperature = config_.temperature;
        request.top_p = config_.top_p;
        request.stream = false;
        
        auto start = std::chrono::steady_clock::now();
        ChatResponse response = engine.chat_completion(request);
        auto end = std::chrono::steady_clock::now();
        
        BenchmarkResult result;
        result.ttft_ms = response.stats.time_to_first_token.count() / 1000.0;
        result.tokens_per_second = response.stats.tokens_per_second();
        result.prompt_processing_tps = response.stats.prompt_processing_tokens_per_second();
        result.prompt_tokens = response.stats.prompt_tokens;
        result.generated_tokens = response.stats.generated_tokens;
        result.peak_memory_mb = response.stats.peak_memory_bytes / (1024.0 * 1024.0);
        result.total_time_ms = response.stats.total_inference_time.count() / 1000.0;
        
        if (!silent && config_.verbose) {
            std::cout << "\n  Prompt: " << prompt.substr(0, 40) << "..." << std::endl;
            std::cout << "  Response: " << response.choices[0].message.substr(0, 80) << "..." << std::endl;
            std::cout << "  TTFT: " << std::fixed << std::setprecision(2) << result.ttft_ms << " ms" << std::endl;
            std::cout << "  Tokens/sec: " << std::setprecision(2) << result.tokens_per_second << std::endl;
            std::cout << "  Prompt tokens: " << result.prompt_tokens << std::endl;
            std::cout << "  Generated tokens: " << result.generated_tokens << std::endl;
            std::cout << "  Peak memory: " << std::setprecision(2) << result.peak_memory_mb << " MB" << std::endl;
            std::cout << std::endl;
        }
        
        return result;
    }
    
    BenchmarkConfig config_;
};


void print_summary(const BenchmarkSummary& summary) {
    std::cout << "== Benchmark Summary ==" << std::endl;
    std::cout << "Runs: " << summary.runs.size() << std::endl;
    std::cout << std::endl;
    
    std::cout << "Time to First Token (TTFT):" << std::endl;
    std::cout << "  Mean:  " << std::fixed << std::setprecision(2) 
              << summary.ttft_mean() << " ms" << std::endl;
    std::cout << "  p50:   " << summary.ttft_p50() << " ms" << std::endl;
    std::cout << "  p95:   " << summary.ttft_p95() << " ms" << std::endl;
    std::cout << "  p99:   " << summary.ttft_p99() << " ms" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Tokens per Second (Throughput):" << std::endl;
    std::cout << "  Mean:  " << std::setprecision(2) << summary.tps_mean() << " tok/s" << std::endl;
    std::cout << "  p50:   " << summary.tps_p50() << " tok/s" << std::endl;
    std::cout << "  p95:   " << summary.tps_p95() << " tok/s" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Peak Memory:" << std::endl;
    std::cout << "  Mean:  " << std::setprecision(2) << summary.memory_mean() << " MB" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    config.model_path = "models/model.gguf";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) config.model_path = argv[++i];
        } else if (arg == "-r" || arg == "--runs") {
            if (i + 1 < argc) config.n_benchmark_runs = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--tokens") {
            if (i + 1 < argc) config.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: arm-ai-benchmark [options]\n"
                      << "  -m, --model <path>   Model path (required)\n"
                      << "  -r, --runs <num>     Number of benchmark runs (default: 5)\n"
                      << "  -t, --tokens <num>   Max tokens per generation (default: 128)\n"
                      << "  -h, --help           Show this help\n";
            return 0;
        }
    }
    
    Benchmark benchmark(config);
    auto summary = benchmark.run();
    print_summary(summary);
    
    return 0;
}
