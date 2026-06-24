#!/bin/bash
# Build MLX for jax-mps.
#
# Usage:
#   ./scripts/setup_deps_mlx.sh [--prefix /path/to/install] [--force]

# shellcheck source=setup_deps_common.sh
source "$(dirname "$0")/setup_deps_common.sh" "$@"

# version.txt holds either a release tag (e.g. v0.31.2) or a full commit SHA
# (to pin an untagged main commit without drift). Both fetch the same way:
# GitHub serves arbitrary commit SHAs to `git fetch` (allowAnySHA1InWant).
MLX_GIT_REF="$(tr -d '[:space:]' < "$REPO_ROOT/third_party/mlx/version.txt")"
if [ -z "$MLX_GIT_REF" ]; then
    echo "Error: MLX Git ref is empty; check $REPO_ROOT/third_party/mlx/version.txt" >&2
    exit 1
fi
MLX_PATCHES_DIR="$REPO_ROOT/third_party/mlx/patches"

echo "=== jax-mps MLX setup ==="
echo "Prefix:       $PREFIX"
echo "Jobs:         $JOBS"
echo "MLX:          $MLX_GIT_REF"
echo ""

if [ "$FORCE_REBUILD" = true ]; then
    rm -rf "$PREFIX/share/cmake/MLX" "$PREFIX/lib/libmlx.a"
    rm -f "$PREFIX/.mlx-tag"
    rm -rf "$BUILD_DIR/mlx-build"
fi

MLX_DIR="$BUILD_DIR/mlx"
MLX_BUILD_DIR="$BUILD_DIR/mlx-build"
MLX_STAMP="$PREFIX/.mlx-tag"
# Include patch checksums in the stamp so patches changes trigger a rebuild.
MLX_PATCHES_HASH=""
if [ -d "$MLX_PATCHES_DIR" ] && ls "$MLX_PATCHES_DIR"/*.patch &>/dev/null; then
    MLX_PATCHES_HASH=$(cat "$MLX_PATCHES_DIR"/*.patch | shasum -a 256 | cut -d' ' -f1)
fi
MLX_FULL_TAG="${MLX_GIT_REF}:${MLX_PATCHES_HASH}"
INSTALLED_MLX_TAG=""
if [ -f "$MLX_STAMP" ]; then
    INSTALLED_MLX_TAG="$(cat "$MLX_STAMP")"
fi
if [ "$INSTALLED_MLX_TAG" != "$MLX_FULL_TAG" ]; then
    # Fetch the source as a hash-pinned tarball rather than cloning. GitHub
    # serves a deterministic archive of any commit's tree at the URL below, so
    # there is no .git directory for a reaped /tmp to leave half-populated (the
    # failure mode that a shallow clone hits when its objects are GC'd).
    MLX_TARBALL="$BUILD_DIR/mlx-${MLX_GIT_REF}.tar.gz"
    echo "=== Downloading MLX source for ref $MLX_GIT_REF ==="
    if [ ! -f "$MLX_TARBALL" ]; then
        mkdir -p "$BUILD_DIR"
        # Download to a temp name then rename so an interrupted download never
        # leaves a truncated tarball that looks complete on the next run.
        curl -fL "https://github.com/ml-explore/mlx/archive/${MLX_GIT_REF}.tar.gz" \
            -o "${MLX_TARBALL}.tmp"
        mv "${MLX_TARBALL}.tmp" "$MLX_TARBALL"
    else
        echo "Using cached tarball $MLX_TARBALL"
    fi

    # Always extract into a pristine tree so patches apply cleanly (and a
    # reaped /tmp simply triggers a re-extract from the cached tarball).
    rm -rf "$MLX_DIR"
    mkdir -p "$MLX_DIR"
    tar -xzf "$MLX_TARBALL" -C "$MLX_DIR" --strip-components=1

    echo "=== Applying MLX patches ==="
    # git apply works outside a git repository: it patches the working tree
    # relative to the current directory (default -p1 strips the a/ b/ prefix).
    for patch in "$MLX_PATCHES_DIR"/*.patch; do
        [ -f "$patch" ] && (cd "$MLX_DIR" && git apply --verbose "$patch")
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
    echo "$MLX_FULL_TAG" > "$MLX_STAMP"
    echo "MLX installed to $PREFIX"
else
    echo "=== MLX already installed ($MLX_GIT_REF) ==="
fi

echo ""
echo "=== MLX setup complete ==="
echo "Installed to: $PREFIX"
