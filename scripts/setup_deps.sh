#!/bin/bash
# Setup script for jax-mps dependencies (LLVM/MLIR + StableHLO)
# These are built once and installed to a prefix directory.
#
# Usage:
#   ./scripts/setup_deps.sh [--prefix /path/to/install] [--force]
#
# Options:
#   --prefix PATH   Install location (default: $HOME/.local/jax-mps-deps)
#   --jobs N        Number of parallel jobs (default: number of CPUs)
#   --force         Force rebuild even if already installed
#
# Default prefix: $HOME/.local/jax-mps-deps

set -e

# Configuration
PREFIX="${PREFIX:-$HOME/.local/jax-mps-deps}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
BUILD_DIR="${BUILD_DIR:-/tmp/jax-mps-deps-build}"

# Pin to versions matching jaxlib 0.9.0 for bytecode compatibility
# These are extracted from XLA commit bb760b047bdbfeff962f0366ad5cc782c98657e0
XLA_COMMIT="${XLA_COMMIT:-bb760b047bdbfeff962f0366ad5cc782c98657e0}"
STABLEHLO_COMMIT="${STABLEHLO_COMMIT:-127d2f238010589ac96f2f402a27afc9dccbb7ab}"
LLVM_COMMIT_OVERRIDE="${LLVM_COMMIT_OVERRIDE:-f6d0a512972a74ef100723b9526a6a0ddb23f894}"

# Abseil and Protobuf versions (protobuf depends on abseil)
ABSEIL_VERSION="${ABSEIL_VERSION:-20250127.0}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-29.3}"

# Parse arguments
FORCE_REBUILD=false
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
        --force)
            FORCE_REBUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If force rebuild, remove existing installations
if [ "$FORCE_REBUILD" = true ]; then
    echo "=== Force rebuild: removing existing installations ==="
    rm -rf "$PREFIX/lib/cmake/mlir" "$PREFIX/lib/cmake/llvm"
    rm -f "$PREFIX/lib/libStablehloOps.a"
    rm -f "$PREFIX/lib/libprotobuf.a" "$PREFIX/lib/libabsl_base.a"
    rm -rf "$BUILD_DIR/llvm-build" "$BUILD_DIR/stablehlo-build"
    rm -rf "$BUILD_DIR/abseil-build" "$BUILD_DIR/protobuf-build"
fi

echo "=== jax-mps dependency setup ==="
echo "Prefix:       $PREFIX"
echo "Jobs:         $JOBS"
echo "Build dir:    $BUILD_DIR"
echo "Abseil:       $ABSEIL_VERSION"
echo "Protobuf:     $PROTOBUF_VERSION"
echo "XLA:          $XLA_COMMIT"
echo "StableHLO:    $STABLEHLO_COMMIT"
echo "LLVM:         $LLVM_COMMIT_OVERRIDE"
echo "Force:        $FORCE_REBUILD"
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

# Build Abseil (required by protobuf, must be static for wheel distribution)
ABSEIL_DIR="$BUILD_DIR/abseil-cpp"
ABSEIL_BUILD_DIR="$BUILD_DIR/abseil-build"
if [ ! -f "$PREFIX/lib/libabsl_base.a" ]; then
    echo "=== Downloading Abseil $ABSEIL_VERSION ==="
    if [ ! -d "$ABSEIL_DIR" ]; then
        curl -L "https://github.com/abseil/abseil-cpp/archive/refs/tags/$ABSEIL_VERSION.tar.gz" | tar xz -C "$BUILD_DIR"
        mv "$BUILD_DIR/abseil-cpp-$ABSEIL_VERSION" "$ABSEIL_DIR"
    fi

    echo "=== Building Abseil (static) ==="
    cmake -G Ninja -B "$ABSEIL_BUILD_DIR" -S "$ABSEIL_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DABSL_BUILD_TESTING=OFF \
        -DABSL_PROPAGATE_CXX_STD=ON

    cmake --build "$ABSEIL_BUILD_DIR" -j "$JOBS"
    cmake --install "$ABSEIL_BUILD_DIR"
    echo "Abseil installed to $PREFIX"
else
    echo "=== Abseil already installed ==="
fi

# Build Protobuf (must be static for wheel distribution)
PROTOBUF_DIR="$BUILD_DIR/protobuf"
PROTOBUF_BUILD_DIR="$BUILD_DIR/protobuf-build"
if [ ! -f "$PREFIX/lib/libprotobuf.a" ]; then
    echo "=== Downloading Protobuf $PROTOBUF_VERSION ==="
    if [ ! -d "$PROTOBUF_DIR" ]; then
        curl -L "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v$PROTOBUF_VERSION.tar.gz" | tar xz -C "$BUILD_DIR"
        mv "$BUILD_DIR/protobuf-$PROTOBUF_VERSION" "$PROTOBUF_DIR"
    fi

    echo "=== Building Protobuf (static) ==="
    cmake -G Ninja -B "$PROTOBUF_BUILD_DIR" -S "$PROTOBUF_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_ABSL_PROVIDER=package \
        -Dprotobuf_BUILD_PROTOBUF_BINARIES=ON \
        -Dprotobuf_BUILD_PROTOC_BINARIES=ON

    cmake --build "$PROTOBUF_BUILD_DIR" -j "$JOBS"
    cmake --install "$PROTOBUF_BUILD_DIR"
    echo "Protobuf installed to $PREFIX"
else
    echo "=== Protobuf already installed ==="
fi

# Clone StableHLO at pinned commit for jaxlib compatibility
STABLEHLO_DIR="$BUILD_DIR/stablehlo"
if [ ! -d "$STABLEHLO_DIR" ]; then
    echo "=== Cloning StableHLO at commit $STABLEHLO_COMMIT ==="
    mkdir -p "$STABLEHLO_DIR"
    cd "$STABLEHLO_DIR"
    git init
    git remote add origin https://github.com/openxla/stablehlo.git
    git fetch --depth 1 origin "$STABLEHLO_COMMIT"
    git checkout FETCH_HEAD
else
    echo "=== Checking StableHLO commit ==="
    cd "$STABLEHLO_DIR"
    CURRENT_COMMIT=$(git rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$STABLEHLO_COMMIT" ]; then
        echo "=== Updating StableHLO to commit $STABLEHLO_COMMIT ==="
        git fetch --depth 1 origin "$STABLEHLO_COMMIT"
        git checkout FETCH_HEAD
    fi
fi

cd "$STABLEHLO_DIR"
# Use LLVM commit override if set, otherwise read from StableHLO
if [ -n "$LLVM_COMMIT_OVERRIDE" ]; then
    LLVM_COMMIT="$LLVM_COMMIT_OVERRIDE"
    echo "Using LLVM commit override: $LLVM_COMMIT"
else
    LLVM_COMMIT=$(cat build_tools/llvm_version.txt)
    echo "StableHLO requires LLVM commit: $LLVM_COMMIT"
fi

# Clone LLVM - fetch only the specific commit we need
LLVM_DIR="$BUILD_DIR/llvm-project"
if [ ! -d "$LLVM_DIR" ]; then
    echo "=== Fetching LLVM commit $LLVM_COMMIT (minimal clone) ==="
    mkdir -p "$LLVM_DIR"
    cd "$LLVM_DIR"
    git init
    git remote add origin https://github.com/llvm/llvm-project.git
    git fetch --depth 1 origin "$LLVM_COMMIT"
    git checkout FETCH_HEAD
else
    echo "=== LLVM already cloned ==="
    cd "$LLVM_DIR"
    # Check if we have the right commit
    CURRENT_COMMIT=$(git rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$LLVM_COMMIT" ]; then
        echo "=== Fetching LLVM commit $LLVM_COMMIT ==="
        git fetch --depth 1 origin "$LLVM_COMMIT"
        git checkout FETCH_HEAD
    fi
fi

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
if [ ! -f "$PREFIX/lib/libStablehloOps.a" ]; then
    echo "=== Patching StableHLO (disable lit tests) ==="
    # StableHLO's test CMakeLists require LLVM FileCheck which we don't install
    # Wrap the lit test setup in if(TARGET FileCheck) to skip when not available
    for f in "$STABLEHLO_DIR/stablehlo/tests/CMakeLists.txt" \
             "$STABLEHLO_DIR/stablehlo/testdata/CMakeLists.txt" \
             "$STABLEHLO_DIR/stablehlo/conversions/linalg/tests/CMakeLists.txt" \
             "$STABLEHLO_DIR/stablehlo/conversions/tosa/tests/CMakeLists.txt"; do
        if [ -f "$f" ] && ! grep -q "if(TARGET FileCheck)" "$f"; then
            python3 -c "
import re, sys
content = open('$f').read()
pattern = r'(configure_lit_site_cfg\([^)]+\)\s*add_lit_testsuite\([^)]+\)\s*add_dependencies\([^)]+\))'
def wrap(m): return 'if(TARGET FileCheck)\n' + m.group(1) + '\nendif()'
print(re.sub(pattern, wrap, content, flags=re.DOTALL))
" > "$f.tmp" && mv "$f.tmp" "$f"
        fi
    done

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

    # StableHLO doesn't install headers by default, do it manually
    echo "=== Installing StableHLO headers ==="
    mkdir -p "$PREFIX/include/stablehlo/dialect"
    mkdir -p "$PREFIX/include/stablehlo/api"
    cp "$STABLEHLO_DIR/stablehlo/dialect/"*.h "$PREFIX/include/stablehlo/dialect/"
    cp "$STABLEHLO_DIR/stablehlo/api/"*.h "$PREFIX/include/stablehlo/api/"
    # Copy generated tablegen headers
    cp "$STABLEHLO_BUILD_DIR/stablehlo/dialect/"*.inc "$PREFIX/include/stablehlo/dialect/" 2>/dev/null || true

    echo "StableHLO installed to $PREFIX"
else
    echo "=== StableHLO already installed ==="
fi

# Install XLA PJRT headers (only the C API header is needed)
if [ ! -f "$PREFIX/include/xla/pjrt/c/pjrt_c_api.h" ]; then
    XLA_DIR="$BUILD_DIR/xla"
    echo "=== Fetching XLA headers at commit $XLA_COMMIT ==="
    if [ ! -d "$XLA_DIR" ]; then
        mkdir -p "$XLA_DIR"
        cd "$XLA_DIR"
        git init
        git remote add origin https://github.com/openxla/xla.git
        git fetch --depth 1 origin "$XLA_COMMIT"
        git checkout FETCH_HEAD
    fi

    mkdir -p "$PREFIX/include/xla/pjrt/c"
    cp "$XLA_DIR/xla/pjrt/c/pjrt_c_api.h" "$PREFIX/include/xla/pjrt/c/"
    echo "XLA PJRT headers installed to $PREFIX"
else
    echo "=== XLA PJRT headers already installed ==="
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
