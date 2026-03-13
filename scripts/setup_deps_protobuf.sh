#!/bin/bash
# Build Abseil and Protobuf for jax-mps.
#
# Usage:
#   ./scripts/setup_deps_protobuf.sh [--prefix /path/to/install] [--force]

# shellcheck source=setup_deps_common.sh
source "$(dirname "$0")/setup_deps_common.sh" "$@"

# Abseil and Protobuf versions (protobuf depends on abseil)
ABSEIL_VERSION="${ABSEIL_VERSION:-20250127.0}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-29.3}"

echo "=== jax-mps Protobuf setup ==="
echo "Prefix:       $PREFIX"
echo "Jobs:         $JOBS"
echo "Abseil:       $ABSEIL_VERSION"
echo "Protobuf:     $PROTOBUF_VERSION"
echo ""

# Version stamp to detect version changes without --force
PROTOBUF_STAMP="$PREFIX/.protobuf-versions"
PROTOBUF_EXPECTED_STAMP="abseil=$ABSEIL_VERSION protobuf=$PROTOBUF_VERSION"
if [ -f "$PROTOBUF_STAMP" ] && [ "$(cat "$PROTOBUF_STAMP")" != "$PROTOBUF_EXPECTED_STAMP" ]; then
    echo "=== Version mismatch detected, forcing rebuild ==="
    FORCE_REBUILD=true
fi

if [ "$FORCE_REBUILD" = true ]; then
    rm -f "$PREFIX/lib/libabsl_base.a" "$PREFIX/lib/libprotobuf.a"
    rm -f "$PROTOBUF_STAMP"
    rm -rf "$BUILD_DIR/abseil-build" "$BUILD_DIR/protobuf-build"
    rm -rf "$BUILD_DIR/abseil-cpp" "$BUILD_DIR/protobuf"
fi

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

echo "$PROTOBUF_EXPECTED_STAMP" > "$PROTOBUF_STAMP"

echo ""
echo "=== Protobuf setup complete ==="
echo "Installed to: $PREFIX"
