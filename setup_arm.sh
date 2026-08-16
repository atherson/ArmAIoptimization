#!/usr/bin/env bash
# =============================================================================
# setup_arm.sh — Arm AI Inference Server build script
#
# Supports: Ubuntu/Debian (apt), Fedora/RHEL (dnf/yum), Arch (pacman),
#           openSUSE (zypper). Detects Arm64 and enables KleidiAI.
#
# Fixes from code review:
#   [FIX-20] apt-key add replaced with gpg --dearmor (deprecated on
#            Ubuntu 20.04+ / Debian Buster+).
#   [FIX-21] 'git checkout origin/master' replaced with
#            'git checkout -B master origin/master' (unambiguous).
#   [FIX-9]  All if/elif/else/fi blocks verified balanced and correctly
#            indented so bash parses them without error.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step()    { echo -e "\n${GREEN}==>${NC} $1"; }
print_error()   { echo -e "${RED}Error:${NC} $1" >&2; exit 1; }
print_warning() { echo -e "${YELLOW}Warning:${NC} $1"; }
print_info()    { echo -e "${BLUE}Info:${NC} $1"; }

# ---------------------------------------------------------------------------
# OS / architecture detection
# ---------------------------------------------------------------------------
detect_os() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_VERSION_ID="${VERSION_ID:-unknown}"
    else
        OS_ID="unknown"
        OS_VERSION_ID="unknown"
    fi

    ARCH=$(uname -m)
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        IS_ARM=true
        print_info "Detected Arm64 architecture."
    elif [[ "$ARCH" == "x86_64" ]]; then
        IS_ARM=false
        print_info "Detected x86_64 architecture."
    else
        IS_ARM=false
        print_warning "Unknown architecture: $ARCH. Assuming non-Arm."
    fi
}

# ---------------------------------------------------------------------------
# System dependency installation
# ---------------------------------------------------------------------------
install_dependencies() {
    print_step "Installing system dependencies..."

    if command -v apt-get &>/dev/null; then
        PKG_MANAGER="apt"
        PKG_UPDATE="apt-get update"
        PKG_INSTALL="apt-get install -y"
        DEPENDENCIES="build-essential cmake git wget curl \
            libcurl4-openssl-dev pkg-config ninja-build g++ gcc lld \
            gpg software-properties-common lsb-release"
    elif command -v dnf &>/dev/null; then
        PKG_MANAGER="dnf"
        PKG_UPDATE="dnf check-update || true"
        PKG_INSTALL="dnf install -y"
        DEPENDENCIES="gcc-c++ cmake git wget curl libcurl-devel \
            pkgconfig ninja-build gcc lld"
    elif command -v yum &>/dev/null; then
        PKG_MANAGER="yum"
        PKG_UPDATE="yum check-update || true"
        PKG_INSTALL="yum install -y"
        DEPENDENCIES="gcc-c++ cmake git wget curl libcurl-devel \
            pkgconfig ninja-build gcc lld"
    elif command -v pacman &>/dev/null; then
        PKG_MANAGER="pacman"
        PKG_UPDATE="pacman -Sy"
        PKG_INSTALL="pacman -S --noconfirm"
        DEPENDENCIES="base-devel cmake git wget curl gcc lld"
    elif command -v zypper &>/dev/null; then
        PKG_MANAGER="zypper"
        PKG_UPDATE="zypper refresh"
        PKG_INSTALL="zypper install -y"
        DEPENDENCIES="gcc-c++ cmake git wget curl libcurl-devel \
            pkgconfig ninja-build gcc lld"
    else
        print_error "Could not detect a supported package manager. \
Please install: gcc, g++, cmake, git, wget, curl, make, ninja-build manually."
    fi

    print_info "Package manager: $PKG_MANAGER"
    print_info "Packages:        $DEPENDENCIES"

    eval "$PKG_UPDATE"  || print_warning "Package list update failed; continuing."
    # shellcheck disable=SC2086
    eval "$PKG_INSTALL $DEPENDENCIES" || print_error "Dependency installation failed."

    # --- CMake version check and upgrade if needed ---
    # [FIX-9] All if/elif/else/fi blocks are explicit and balanced.
    if command -v cmake &>/dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
        REQUIRED_CMAKE="3.14"

        if [[ "$(printf '%s\n' "$REQUIRED_CMAKE" "$CMAKE_VERSION" \
                    | sort -V | head -n1)" != "$REQUIRED_CMAKE" ]]; then
            print_warning "CMake $CMAKE_VERSION < $REQUIRED_CMAKE. Attempting upgrade..."

            if [[ "$PKG_MANAGER" == "apt" ]]; then
                # [FIX-20] Use gpg --dearmor instead of deprecated apt-key add.
                print_info "Adding Kitware CMake APT repository..."
                wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
                    | gpg --dearmor \
                    | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
                echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
                    | tee /etc/apt/sources.list.d/kitware.list >/dev/null
                apt-get update
                apt-get install -y cmake
            elif [[ "$PKG_MANAGER" == "dnf" || "$PKG_MANAGER" == "yum" ]]; then
                eval "$PKG_INSTALL cmake3" \
                    || print_warning "Could not install cmake3 — continuing."
                if command -v cmake3 &>/dev/null && ! command -v cmake &>/dev/null; then
                    ln -sf "$(command -v cmake3)" /usr/local/bin/cmake || true
                fi
            else
                print_warning "Please manually upgrade CMake to >= $REQUIRED_CMAKE."
            fi  # end inner pkg-manager branch
        else
            print_info "CMake $CMAKE_VERSION meets the >= $REQUIRED_CMAKE requirement."
        fi  # end version comparison
    else
        print_error "CMake not found after installation. Please install manually."
    fi  # end outer cmake detection
}

# ---------------------------------------------------------------------------
# Project directory setup and llama.cpp clone/update
# ---------------------------------------------------------------------------
setup_project() {
    print_step "Setting up project directories..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"

    mkdir -p third_party
    THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"
    LLAMA_CPP_DIR="$THIRD_PARTY_DIR/llama.cpp"

    if [[ -d "$LLAMA_CPP_DIR/.git" ]]; then
        print_step "Updating existing llama.cpp clone..."
        cd "$LLAMA_CPP_DIR"
        git fetch --all --prune
        # [FIX-21] Explicit branch checkout — no ambiguity with remote names.
        git checkout -B master origin/master
        cd "$SCRIPT_DIR"
    else
        print_step "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    fi

    mkdir -p build
    BUILD_DIR="$SCRIPT_DIR/build"
}

# ---------------------------------------------------------------------------
# CMake configuration
# ---------------------------------------------------------------------------
configure_cmake() {
    print_step "Configuring CMake build (${BUILD_TYPE})..."

    cd "$BUILD_DIR"

    CMAKE_FLAGS=(
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DBUILD_SHARED_LIBS=OFF
        -DLLAMA_CURL=OFF
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_SERVER=OFF
    )

    if [[ "$IS_ARM" == true ]]; then
        print_info "Enabling KleidiAI optimisations for Armv9-A..."
        CMAKE_FLAGS+=(
            -DGGML_NATIVE=OFF
            -DGGML_CPU_ARM_ARCH="armv9-a+i8mm+dotprod"
            -DGGML_CPU_KLEIDIAI=ON
            -DGGML_CPU_AARCH64=ON
            -DGGML_AVX=OFF
            -DGGML_AVX2=OFF
            -DGGML_AVX512=OFF
            -DGGML_FMA=OFF
            -DGGML_F16C=OFF
        )
    else
        print_info "Not on Arm — skipping KleidiAI flags."
        CMAKE_FLAGS+=(
            -DGGML_NATIVE=OFF
            -DGGML_AVX=OFF
            -DGGML_AVX2=OFF
            -DGGML_AVX512=OFF
            -DGGML_FMA=OFF
            -DGGML_F16C=OFF
        )
    fi

    cmake "${SCRIPT_DIR}" "${CMAKE_FLAGS[@]}" \
        || print_error "CMake configuration failed."

    if [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
        ln -sf "$BUILD_DIR/compile_commands.json" \
               "$SCRIPT_DIR/compile_commands.json"
        print_info "compile_commands.json symlinked for LSP support."
    fi
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build_project() {
    print_step "Building ($(nproc) parallel jobs)..."
    cd "$BUILD_DIR"
    make -j"$(nproc)" || print_error "Build failed."
    print_info "Build successful."
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}Setup complete!${NC}"
    echo "============================================================"
    echo ""
    echo "Binaries:"
    echo "  Server:    $BUILD_DIR/arm-ai-server"
    echo "  Benchmark: $BUILD_DIR/arm-ai-benchmark"
    echo "  Tests:     $BUILD_DIR/arm-ai-tests"
    echo ""
    echo "Quick start:"
    echo "  $BUILD_DIR/arm-ai-server    -m /path/to/model.gguf -p 8080"
    echo "  $BUILD_DIR/arm-ai-benchmark -m /path/to/model.gguf -r 5"
    echo "  $BUILD_DIR/arm-ai-tests"
    echo ""
    echo "ASan build:"
    echo "  cmake -DCMAKE_BUILD_TYPE=Debug \\"
    echo "        -DCMAKE_CXX_COMPILER=clang++ \\"
    echo "        -DCMAKE_CXX_FLAGS=\"-fsanitize=address,undefined -g\" \\"
    echo "        -DCMAKE_EXE_LINKER_FLAGS=\"-fsanitize=address,undefined\" \\"
    echo "        ../  &&  make -j\$(nproc)"
    echo ""
    if [[ "$IS_ARM" == true ]]; then
        echo "KleidiAI optimisations: ENABLED"
        echo "Target architecture:    armv9-a+i8mm+dotprod"
    else
        echo "KleidiAI optimisations: DISABLED (not on Arm64)"
    fi
    echo "============================================================"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BUILD_TYPE="Release"
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "  --debug       Build in Debug mode"
            echo "  --skip-deps   Skip installing system dependencies"
            echo "  --help, -h    Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    detect_os

    if [[ "$SKIP_DEPS" == false ]]; then
        install_dependencies
    else
        print_info "Skipping dependency installation (--skip-deps)."
    fi

    setup_project
    configure_cmake
    build_project
    print_summary
}

main
