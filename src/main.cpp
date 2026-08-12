#include "server.hpp"

#include <iostream>
#include <csignal>
#include <atomic>

std::atomic<bool> shutdown_flag{false};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    shutdown_flag = true;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -m, --model <path>    Path to GGUF model file (required)\n"
              << "  -p, --port <port>     HTTP server port (default: 8080)\n"
              << "  -c, --context <size>  Context window size (default: 4096)\n"
              << "  -t, --threads <num>   Number of CPU threads (default: auto)\n"
              << "  -h, --help            Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    int port = 8080;
    int n_ctx = 4096;
    int n_threads = -1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) port = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--context") {
            if (i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) n_threads = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: Model path is required. Use -m to specify." << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "=== Arm AI Optimization Challenge 2026 ===" << std::endl;
    std::cout << "Starting Arm AI Server..." << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    std::cout << "  Port: " << port << std::endl;
    std::cout << "  Context: " << n_ctx << std::endl;
    std::cout << "  Threads: " << (n_threads > 0 ? std::to_string(n_threads) : "auto") << std::endl;
    std::cout << std::endl;

    arm_ai::AIServer server;
    
    if (!server.start(model_path, port, n_ctx, n_threads)) {
        std::cerr << "Failed to start server." << std::endl;
        return 1;
    }

    std::cout << "Server started successfully!" << std::endl;
    std::cout << "OpenAI-compatible API available at: http://localhost:" << port << "/v1/" << std::endl;
    std::cout << "  POST /v1/chat/completions" << std::endl;
    std::cout << "  GET  /v1/models" << std::endl;
    std::cout << "  GET  /health" << std::endl;
    std::cout << std::endl;
    std::cout << "Press Ctrl+C to stop..." << std::endl;

    while (!shutdown_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Stopping server..." << std::endl;
    server.stop();
    std::cout << "Server stopped. Goodbye!" << std::endl;

    return 0;
}
