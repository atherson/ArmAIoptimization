FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libcurl4-openssl-dev \
    pkg-config \
    ninja-build \
    gcc \
    g++ \
    lld \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the project files
COPY . .

# Build the project
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -j$(nproc)

# Expose the server port
EXPOSE 8080

# Run the server
CMD ["./build/arm-ai-server", "-m", "/models/model.gguf", "-p", "8080"]
