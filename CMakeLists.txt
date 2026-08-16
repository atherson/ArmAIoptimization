cmake_minimum_required(VERSION 3.14)
project(
  optimization
  VERSION 1.0.0
  LANGUAGES C CXX)

# =============================================================================
# Global settings
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# [FIX-17] -O3 and -flto removed from global add_compile_options. They are
# applied per-target below so that FetchContent dependencies (llama.cpp, gtest)
# are not forced into our LTO domain — LTO across independently built static
# libraries can cause subtle link failures.

# [FIX-19] llama.cpp pinned to a specific release tag for reproducible builds.
# Update this SHA when upgrading llama.cpp intentionally.
set(LLAMA_CPP_GIT_TAG
    "b5598"
    CACHE STRING "llama.cpp git tag / commit SHA to fetch (default: b5598)")

# =============================================================================
# llama.cpp configuration — set before FetchContent so the sub-project picks
# them up during its own CMakeLists processing.
# =============================================================================

set(GGML_NATIVE
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS
    OFF
    CACHE BOOL "" FORCE)
set(LLAMA_CURL
    OFF
    CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES
    OFF
    CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER
    OFF
    CACHE BOOL "" FORCE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # Let llama.cpp's CMake handle the march for ggml internals via the cache
  # variable below. We apply our own -march only to our targets further down.
  # [FIX-18]
  set(GGML_CPU_ARM_ARCH
      "armv9-a+sve+i8mm+dotprod"
      CACHE STRING "" FORCE)
  set(GGML_CPU_KLEIDIAI
      ON
      CACHE BOOL "" FORCE)
  set(GGML_CPU_AARCH64
      ON
      CACHE BOOL "" FORCE)
  set(GGML_AVX
      OFF
      CACHE BOOL "" FORCE)
  set(GGML_AVX2
      OFF
      CACHE BOOL "" FORCE)
  set(GGML_AVX512
      OFF
      CACHE BOOL "" FORCE)
  set(GGML_FMA
      OFF
      CACHE BOOL "" FORCE)
  set(GGML_F16C
      OFF
      CACHE BOOL "" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(GGML_AVX
      ON
      CACHE BOOL "" FORCE)
  set(GGML_AVX2
      ON
      CACHE BOOL "" FORCE)
  set(GGML_FMA
      ON
      CACHE BOOL "" FORCE)
  set(GGML_F16C
      ON
      CACHE BOOL "" FORCE)
  set(GGML_AVX512
      OFF
      CACHE BOOL "" FORCE)
else()
  set(GGML_AVX
      OFF
      CACHE BOOL "" FORCE)
  set(GGML_AVX2
      OFF
      CACHE BOOL "" FORCE)
endif()

# =============================================================================
# Fetch dependencies
# =============================================================================

include(FetchContent)

# [FIX-19] Pinned tag — update intentionally, not via 'master'.
FetchContent_Declare(
  llama.cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG ${LLAMA_CPP_GIT_TAG}
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp)

FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2)

set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama.cpp httplib json googletest)

# =============================================================================
# Per-target compile flags helper [FIX-17] [FIX-18] Optimisation and march flags
# scoped to OUR targets only.
# =============================================================================

# Collect our optimisation flags into a list so they can be reused.
set(OWN_COMPILE_FLAGS -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # [FIX-18] -march for our source files only; llama.cpp is already configured
  # via GGML_CPU_ARM_ARCH above.
  list(APPEND OWN_COMPILE_FLAGS -march=armv9-a+sve2 -mtune=cortex-a710)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  list(APPEND OWN_COMPILE_FLAGS -march=native)
endif()

# LTO flags (link-time optimisation for our targets only). [FIX-17]
set(OWN_LTO_COMPILE_FLAGS -flto)
set(OWN_LTO_LINK_FLAGS -flto)

# =============================================================================
# Targets
# =============================================================================

# ---------------------------------------------------------------------------
# arm-ai-server
# ---------------------------------------------------------------------------
add_executable(arm-ai-server src/main.cpp src/inference_engine.cpp
                             src/server.cpp)

target_include_directories(
  arm-ai-server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${llama.cpp_SOURCE_DIR}/include
          ${httplib_SOURCE_DIR} ${json_SOURCE_DIR}/include)

target_link_libraries(
  arm-ai-server PRIVATE llama httplib::httplib nlohmann_json::nlohmann_json
                        pthread)

target_compile_options(arm-ai-server PRIVATE ${OWN_COMPILE_FLAGS}
                                             ${OWN_LTO_COMPILE_FLAGS})

target_link_options(arm-ai-server PRIVATE ${OWN_LTO_LINK_FLAGS})

# ---------------------------------------------------------------------------
# arm-ai-benchmark
# ---------------------------------------------------------------------------
add_executable(arm-ai-benchmark src/benchmark.cpp src/inference_engine.cpp)

target_include_directories(
  arm-ai-benchmark
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${llama.cpp_SOURCE_DIR}/include
          ${json_SOURCE_DIR}/include)

target_link_libraries(arm-ai-benchmark
                      PRIVATE llama nlohmann_json::nlohmann_json pthread)

target_compile_options(arm-ai-benchmark PRIVATE ${OWN_COMPILE_FLAGS}
                                                ${OWN_LTO_COMPILE_FLAGS})

target_link_options(arm-ai-benchmark PRIVATE ${OWN_LTO_LINK_FLAGS})

# ---------------------------------------------------------------------------
# arm-ai-tests ASan/TSan builds: override OWN_COMPILE_FLAGS via cmake
# -DCMAKE_CXX_FLAGS=... See README for exact flags.
# ---------------------------------------------------------------------------
enable_testing()

add_executable(arm-ai-tests src/test.cpp src/inference_engine.cpp)

target_include_directories(
  arm-ai-tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${llama.cpp_SOURCE_DIR}/include
          ${httplib_SOURCE_DIR} ${json_SOURCE_DIR}/include)

target_link_libraries(
  arm-ai-tests PRIVATE llama GTest::gtest_main httplib::httplib
                       nlohmann_json::nlohmann_json pthread)

# Tests built at -O1 with debug info for better sanitiser traces.
target_compile_options(arm-ai-tests PRIVATE -O1 -g)

include(GoogleTest)
gtest_discover_tests(arm-ai-tests)

# =============================================================================
# ASan / TSan helper targets (convenience, not built by default)
# =============================================================================

# To build with ASan: cmake -DCMAKE_BUILD_TYPE=Debug \
# -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
# -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
# -DCMAKE_CXX_COMPILER=clang++ ..
#
# To build with TSan (catches the SSE mutex/network race): cmake
# -DCMAKE_BUILD_TYPE=Debug \ -DCMAKE_CXX_FLAGS="-fsanitize=thread
# -fno-omit-frame-pointer -g" \ -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" ..
#
# Valgrind (no ASan): valgrind --leak-check=full --track-origins=yes
# --show-leak-kinds=all \ ./arm-ai-server -m models/model.gguf -p 8080
