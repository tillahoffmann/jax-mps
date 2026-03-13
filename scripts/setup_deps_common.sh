#!/bin/bash
# Common setup for jax-mps dependency build scripts.
# Source this file from component scripts; do not execute directly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX="${PREFIX:-$HOME/.local/jax-mps-deps}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
BUILD_DIR="${BUILD_DIR:-/tmp/jax-mps-deps-build}"
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix) PREFIX="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --force) FORCE_REBUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$PREFIX" "$BUILD_DIR"

for tool in cmake ninja git; do
    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is required but not installed"
        echo "On macOS: brew install cmake ninja"
        exit 1
    fi
done
