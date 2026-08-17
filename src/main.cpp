#include "server.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

static std::atomic<bool> shutdown_flag{false};

static void signal_handler(int sig) {
  std::cout << "\nReceived signal " << sig << ", shutting down...\n";
  shutdown_flag.store(true, std::memory_order_release);
}

static void print_usage(const char *prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  -m, --model <path>    Path to GGUF model file (required)\n"
            << "  -p, --port <port>     HTTP server port (default: 8080)\n"
            << "  -c, --context <size>  Context window size (default: 4096)\n"
            << "  -t, --threads <num>   Number of CPU threads (default: auto)\n"
            << "  -h, --help            Show this help message\n\n";
}

int main(int argc, char *argv[]) {
  std::string model_path;
  int port = 8080;
  int n_ctx = 4096;
  int n_threads = -1;

  try {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if ((arg == "-m" || arg == "--model") && i + 1 < argc)
        model_path = argv[++i];
      else if ((arg == "-p" || arg == "--port") && i + 1 < argc)
        port = std::stoi(argv[++i]);
      else if ((arg == "-c" || arg == "--context") && i + 1 < argc)
        n_ctx = std::stoi(argv[++i]);
      else if ((arg == "-t" || arg == "--threads") && i + 1 < argc)
        n_threads = std::stoi(argv[++i]);
      else if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        return 0;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Error parsing CLI arguments: Invalid numeric value.\n";
    return 1;
  }

  if (model_path.empty()) {
    std::cerr << "Error: model path is required. Use -m to specify.\n\n";
    print_usage(argv[0]);
    return 1;
  }

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  arm_ai::AIServer server;

  if (!server.start(model_path, port, n_ctx, n_threads)) {
    std::cerr << "Failed to start server.\n";
    return 1;
  }

  std::cout << "Server started successfully on port " << port
            << "!\nPress Ctrl+C to stop...\n";

  while (!shutdown_flag.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  server.stop();
  return 0;
}
