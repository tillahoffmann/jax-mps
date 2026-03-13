#!/bin/bash
# Setup all jax-mps dependencies (LLVM/MLIR + StableHLO + Protobuf + MLX).
#
# This script calls the individual component scripts sequentially. For CI,
# the component scripts are called directly in parallel jobs with independent
# caches. See .github/workflows/build.yml.
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/setup_deps_protobuf.sh" "$@"
"$SCRIPT_DIR/setup_deps_llvm.sh" "$@"
"$SCRIPT_DIR/setup_deps_mlx.sh" "$@"

PREFIX="${PREFIX:-$HOME/.local/jax-mps-deps}"
prev=""
for arg in "$@"; do
    if [ "$prev" = "--prefix" ]; then PREFIX="$arg"; fi
    prev="$arg"
done

echo ""
echo "=== All dependencies installed ==="
echo ""
echo "To build jax-mps, use:"
echo "  cmake -B build -DCMAKE_PREFIX_PATH=$PREFIX"
echo "  cmake --build build"
echo ""
echo "Or set environment variable:"
echo "  export CMAKE_PREFIX_PATH=$PREFIX"
