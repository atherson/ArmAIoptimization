cmake_minimum_required(VERSION 3.14)
project(arm_ai_challenge VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv9-a+i8mm+dotprod" CACHE STRING "" FORCE)
set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)

# Optional: additional Arm-specific flags
set(GGML_CPU_AARCH64 ON CACHE BOOL "" FORCE)

# Disable x86-specific optimizations
set(GGML_AVX OFF CACHE BOOL "" FORCE)
set(GGML_AVX2 OFF CACHE BOOL "" FORCE)
set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
set(GGML_FMA OFF CACHE BOOL "" FORCE)
set(GGML_F16C OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
    llama.cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG origin/master  
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)  

FetchContent_MakeAvailable(llama.cpp)

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

add_executable(arm-ai-server
    src/main.cpp
    src/inference_engine.cpp
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

# Pass compile definitions for Arm optimization
target_compile_definitions(arm-ai-server PRIVATE
    GGML_CPU_KLEIDIAI=1
)


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
