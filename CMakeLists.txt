cmake_minimum_required(VERSION 3.14)
project(optimization VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Global optimisation flags
add_compile_options(-O3 -flto)

# -------------------------------------------------------------------
# Arm architecture optimisations (also applied to your own source)
# -------------------------------------------------------------------
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_options(-march=armv9-a+sve2 -mtune=cortex-a710)
endif()

# -------------------------------------------------------------------
# Configure llama.cpp for Arm with KleidiAI
# -------------------------------------------------------------------
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv9-a+sve+i8mm+dotprod" CACHE STRING "" FORCE)
set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
set(GGML_CPU_AARCH64 ON CACHE BOOL "" FORCE)

# Disable x86-specific optimisations (they won't be used anyway)
set(GGML_AVX OFF CACHE BOOL "" FORCE)
set(GGML_AVX2 OFF CACHE BOOL "" FORCE)
set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
set(GGML_FMA OFF CACHE BOOL "" FORCE)
set(GGML_F16C OFF CACHE BOOL "" FORCE)

# -------------------------------------------------------------------
# Fetch llama.cpp – use 'master' branch or a specific commit SHA
# -------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    llama.cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    # Use 'master' for latest, or replace with a fixed commit SHA (e.g., "9f3b9f1c7d3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c")
    GIT_TAG master
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
)

# Disable unnecessary parts of llama.cpp
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama.cpp)

# -------------------------------------------------------------------
# Other dependencies
# -------------------------------------------------------------------
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.3
)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# -------------------------------------------------------------------
# Main executable: arm-ai-server
# -------------------------------------------------------------------
add_executable(arm-ai-server
    src/main.cpp
    src/inference_engine.cpp
    src/server.cpp
)

target_include_directories(arm-ai-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${llama.cpp_SOURCE_DIR}/include
    ${httplib_SOURCE_DIR}
    ${json_SOURCE_DIR}/include
)

target_link_libraries(arm-ai-server PRIVATE
    llama
    httplib::httplib
    nlohmann_json::nlohmann_json
    pthread
)

# -------------------------------------------------------------------
# Benchmark executable
# -------------------------------------------------------------------
add_executable(arm-ai-benchmark
    src/benchmark.cpp
    src/inference_engine.cpp
)

target_include_directories(arm-ai-benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${llama.cpp_SOURCE_DIR}/include
    ${json_SOURCE_DIR}/include
)

target_link_libraries(arm-ai-benchmark PRIVATE
    llama
    nlohmann_json::nlohmann_json
    pthread
)

# -------------------------------------------------------------------
# Test executable (using GoogleTest)
# -------------------------------------------------------------------
enable_testing()

add_executable(arm-ai-tests
    src/test.cpp
    src/inference_engine.cpp
)

target_include_directories(arm-ai-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${llama.cpp_SOURCE_DIR}/include
    ${httplib_SOURCE_DIR}
    ${json_SOURCE_DIR}/include
)

target_link_libraries(arm-ai-tests PRIVATE
    llama
    GTest::gtest_main
    httplib::httplib
    nlohmann_json::nlohmann_json
    pthread
)

include(GoogleTest)
gtest_discover_tests(arm-ai-tests)
