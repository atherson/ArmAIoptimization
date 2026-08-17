set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm64)

# Target-specific compiler flags for Armv9-A with KleidiAI acceleration
set(ARM_FLAGS "-march=armv9-a+sve2+i8mm+dotprod -O3 -ffast-math -flto")
set(CMAKE_C_FLAGS
    "${ARM_FLAGS}"
    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS
    "${ARM_FLAGS}"
    CACHE STRING "" FORCE)

# Enable llama.cpp / GGML KleidiAI integrations
set(GGML_CPU_KLEIDIAI
    ON
    CACHE BOOL "" FORCE)
set(GGML_NATIVE
    OFF
    CACHE BOOL "" FORCE)
