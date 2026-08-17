cmake_minimum_required(VERSION 3.18)
project(arm_ai_inference CXX C)

# Force C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release build if unspecified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# System Dependencies
find_package(Threads REQUIRED)

# ------------------------------------------------------------------------------
# Dependencies Integration Assumes third-party headers/libraries are available
# in system paths or submodules
# ------------------------------------------------------------------------------

# 1. llama.cpp integration
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
  add_subdirectory(third_party/llama.cpp)
else()
  find_package(llama REQUIRED)
endif()

# 1. Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ------------------------------------------------------------------------------
# Core Engine Shared Library
# ------------------------------------------------------------------------------
add_library(
  arm_ai_core STATIC src/inference_engine.cpp src/inference_engine.hpp
                     src/server.cpp src/server.hpp src/benchmark_helper.hpp)

target_link_libraries(arm_ai_core PUBLIC llama Threads::Threads)

# Enable compiler warnings
target_compile_options(arm_ai_core PRIVATE -Wall -Wextra -Wpedantic)

# ------------------------------------------------------------------------------
# Executable 1: HTTP AI Server
# ------------------------------------------------------------------------------
add_executable(arm-ai-server src/main.cpp)

target_link_libraries(arm-ai-server PRIVATE arm_ai_core)

# ------------------------------------------------------------------------------
# Executable 2: Benchmarking CLI Utility
# ------------------------------------------------------------------------------
add_executable(arm-ai-benchmark src/benchmark.cpp)

target_link_libraries(arm-ai-benchmark PRIVATE arm_ai_core)

# Set runtime output directories for built binaries
set_target_properties(arm-ai-server arm-ai-benchmark
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
