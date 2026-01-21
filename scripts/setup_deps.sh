#!/bin/bash
# Setup script for jax-mps dependencies (LLVM/MLIR + StableHLO)
# These are built once and installed to a prefix directory.
#
# Usage:
#   ./scripts/setup_deps.sh [--prefix /path/to/install]
#
# Default prefix: $HOME/.local/jax-mps-deps

set -e

# Configuration
PREFIX="${PREFIX:-$HOME/.local/jax-mps-deps}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
BUILD_DIR="${BUILD_DIR:-/tmp/jax-mps-deps-build}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== jax-mps dependency setup ==="
echo "Prefix:    $PREFIX"
echo "Jobs:      $JOBS"
echo "Build dir: $BUILD_DIR"
echo ""

mkdir -p "$PREFIX"
mkdir -p "$BUILD_DIR"

# Check for required tools
for tool in cmake ninja git; do
    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is required but not installed"
        echo "On macOS: brew install cmake ninja"
        exit 1
    fi
done

# Clone StableHLO (includes reference to required LLVM version)
STABLEHLO_DIR="$BUILD_DIR/stablehlo"
if [ ! -d "$STABLEHLO_DIR" ]; then
    echo "=== Cloning StableHLO ==="
    git clone https://github.com/openxla/stablehlo.git "$STABLEHLO_DIR"
else
    echo "=== Updating StableHLO ==="
    cd "$STABLEHLO_DIR" && git fetch origin && git checkout main && git pull
fi

cd "$STABLEHLO_DIR"
LLVM_COMMIT=$(cat build_tools/llvm_version.txt)
echo "StableHLO requires LLVM commit: $LLVM_COMMIT"

# Clone/update LLVM
LLVM_DIR="$BUILD_DIR/llvm-project"
if [ ! -d "$LLVM_DIR" ]; then
    echo "=== Cloning LLVM (this may take a while) ==="
    git clone --filter=blob:none https://github.com/llvm/llvm-project.git "$LLVM_DIR"
fi

echo "=== Checking out LLVM commit $LLVM_COMMIT ==="
cd "$LLVM_DIR"
git fetch origin "$LLVM_COMMIT"
git checkout "$LLVM_COMMIT"

# Build LLVM/MLIR
LLVM_BUILD_DIR="$BUILD_DIR/llvm-build"
if [ ! -f "$PREFIX/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "=== Building LLVM/MLIR ==="
    cmake -G Ninja -B "$LLVM_BUILD_DIR" -S "$LLVM_DIR/llvm" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DLLVM_ENABLE_ZSTD=OFF \
        -DLLVM_ENABLE_ZLIB=OFF

    cmake --build "$LLVM_BUILD_DIR" -j "$JOBS"
    cmake --install "$LLVM_BUILD_DIR"
    echo "LLVM/MLIR installed to $PREFIX"
else
    echo "=== LLVM/MLIR already installed ==="
fi

# Build StableHLO
STABLEHLO_BUILD_DIR="$BUILD_DIR/stablehlo-build"
if [ ! -f "$PREFIX/lib/cmake/stablehlo/StablehloConfig.cmake" ]; then
    echo "=== Building StableHLO ==="
    cmake -G Ninja -B "$STABLEHLO_BUILD_DIR" -S "$STABLEHLO_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DMLIR_DIR="$PREFIX/lib/cmake/mlir" \
        -DLLVM_DIR="$PREFIX/lib/cmake/llvm" \
        -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
        -DSTABLEHLO_BUILD_EMBEDDED=OFF

    cmake --build "$STABLEHLO_BUILD_DIR" -j "$JOBS"
    cmake --install "$STABLEHLO_BUILD_DIR"
    echo "StableHLO installed to $PREFIX"
else
    echo "=== StableHLO already installed ==="
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Dependencies installed to: $PREFIX"
echo ""
echo "To build jax-mps, use:"
echo "  cmake -B build -DCMAKE_PREFIX_PATH=$PREFIX"
echo "  cmake --build build"
echo ""
echo "Or set environment variable:"
echo "  export CMAKE_PREFIX_PATH=$PREFIX"
