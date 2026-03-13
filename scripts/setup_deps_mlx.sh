#!/bin/bash
# Build MLX for jax-mps.
#
# Usage:
#   ./scripts/setup_deps_mlx.sh [--prefix /path/to/install] [--force]

# shellcheck source=setup_deps_common.sh
source "$(dirname "$0")/setup_deps_common.sh" "$@"

MLX_GIT_TAG="$(tr -d '[:space:]' < "$REPO_ROOT/third_party/mlx/version.txt")"
if [ -z "$MLX_GIT_TAG" ]; then
    echo "Error: MLX Git tag is empty; check $REPO_ROOT/third_party/mlx/version.txt" >&2
    exit 1
fi
MLX_PATCHES_DIR="$REPO_ROOT/third_party/mlx/patches"

echo "=== jax-mps MLX setup ==="
echo "Prefix:       $PREFIX"
echo "Jobs:         $JOBS"
echo "MLX:          $MLX_GIT_TAG"
echo ""

if [ "$FORCE_REBUILD" = true ]; then
    rm -rf "$PREFIX/share/cmake/MLX" "$PREFIX/lib/libmlx.a"
    rm -f "$PREFIX/.mlx-tag"
    rm -rf "$BUILD_DIR/mlx-build"
fi

MLX_DIR="$BUILD_DIR/mlx"
MLX_BUILD_DIR="$BUILD_DIR/mlx-build"
MLX_STAMP="$PREFIX/.mlx-tag"
INSTALLED_MLX_TAG=""
if [ -f "$MLX_STAMP" ]; then
    INSTALLED_MLX_TAG="$(cat "$MLX_STAMP")"
fi
if [ "$INSTALLED_MLX_TAG" != "$MLX_GIT_TAG" ]; then
    echo "=== Cloning MLX at tag $MLX_GIT_TAG ==="
    if [ ! -d "$MLX_DIR" ]; then
        mkdir -p "$MLX_DIR"
        cd "$MLX_DIR"
        git init
        git remote add origin https://github.com/ml-explore/mlx.git
    else
        cd "$MLX_DIR"
    fi
    git fetch --depth 1 origin tag "$MLX_GIT_TAG" --no-tags
    git checkout FETCH_HEAD

    echo "=== Applying MLX patches ==="
    git checkout -- . && git clean -fd
    for patch in "$MLX_PATCHES_DIR"/*.patch; do
        [ -f "$patch" ] && git apply --verbose "$patch"
    done

    echo "=== Building MLX (static) ==="
    cmake -G Ninja -B "$MLX_BUILD_DIR" -S "$MLX_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DMLX_BUILD_TESTS=OFF \
        -DMLX_BUILD_EXAMPLES=OFF \
        -DMLX_BUILD_BENCHMARKS=OFF \
        -DMLX_BUILD_PYTHON_BINDINGS=OFF

    cmake --build "$MLX_BUILD_DIR" -j "$JOBS"
    cmake --install "$MLX_BUILD_DIR"
    echo "$MLX_GIT_TAG" > "$MLX_STAMP"
    echo "MLX installed to $PREFIX"
else
    echo "=== MLX already installed ($MLX_GIT_TAG) ==="
fi

echo ""
echo "=== MLX setup complete ==="
echo "Installed to: $PREFIX"
