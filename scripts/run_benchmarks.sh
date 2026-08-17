#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
MODEL_PATH="${1:-models/llama-3-8b-instruct.Q4_K_M.gguf}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' not found."
    exit 1
fi

echo "==> Configuring release build..."
cmake -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=cmake/armv9-toolchain.cmake

echo "==> Compiling target binaries..."
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

echo "==> Running benchmark pinned to performance cores..."
taskset -c 4-7 ./"$BUILD_DIR"/arm-ai-benchmark \
    --model "$MODEL_PATH" \
    --threads 4 \
    --tokens 128 \
    --runs 10
