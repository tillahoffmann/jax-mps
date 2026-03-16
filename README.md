# jax-mps [![GitHub Action Badge](https://github.com/tillahoffmann/jax-mps/actions/workflows/build.yml/badge.svg)](https://github.com/tillahoffmann/jax-mps/actions/workflows/build.yml) [![PyPI](https://img.shields.io/pypi/v/jax-mps)](https://pypi.org/project/jax-mps/) ![JAX tests](https://img.shields.io/badge/JAX_tests-85.7%25_passing-yellow)[^1]

A JAX backend for Apple Silicon using [MLX](https://github.com/ml-explore/mlx), enabling GPU-accelerated JAX computations on Mac.

> [!NOTE]
> Our CI currently only validates that the project compiles because GitHub's hosted runners don't have access to Apple GPUs. If you have a Mac (e.g., a Mac Mini) that could serve as a [self-hosted GitHub Actions runner](https://docs.github.com/en/actions/hosting-your-own-runners) for this project, please [open an issue](https://github.com/tillahoffmann/jax-mps/issues) — it would let us run the full test suite on every PR and help us move much faster.

## Example

jax-mps achieves a ~3.7x speed-up over the CPU backend when training a simple ResNet18 model on CIFAR-10 using an M4 MacBook Air.

```bash
$ JAX_PLATFORMS=cpu uv run examples/resnet/main.py --steps=30
loss = 0.029: 100%|██████████| 30/30 [01:41<00:00,  3.37s/it]
Final training loss: 0.029
Time per step (second half): 3.437

$ JAX_PLATFORMS=mps uv run examples/resnet/main.py --steps=30
WARNING:...:jax._src.xla_bridge:905: Platform 'mps' is experimental and not all JAX functionality may be correctly supported!
loss = 0.029: 100%|██████████| 30/30 [00:27<00:00,  1.07it/s]
Final training loss: 0.029
Time per step (second half): 0.928
```

## Installation

jax-mps requires macOS on Apple Silicon and Python 3.13. Install it with pip:

```bash
pip install jax-mps
```

The plugin registers itself with JAX automatically and is enabled by default. Set `JAX_PLATFORMS=mps` to select the MPS backend explicitly.

jax-mps is built against the StableHLO bytecode format matching jaxlib 0.9.x. Using a different jaxlib version will likely cause deserialization failures at JIT compile time. See [Version Pinning](#version-pinning) for details.

## Architecture

This project implements a [PJRT plugin](https://openxla.org/xla/pjrt) that uses [MLX](https://github.com/ml-explore/mlx) to execute JAX programs on Apple Silicon GPUs. The evaluation proceeds in several stages:

1. The JAX program is lowered to [StableHLO](https://openxla.org/stablehlo), a set of high-level operations for machine learning applications.
2. The plugin parses the StableHLO representation and maps operations to MLX equivalents. Compiled programs are cached to avoid re-parsing on repeated invocations.
3. The MLX operations are executed on the GPU and results are returned to the caller.

## Building

1. Install build tools and build and install LLVM/MLIR & StableHLO. This is a one-time setup and takes about 30 minutes. See the `setup_deps.sh` script for further options, such as forced re-installation, installation location, etc. The script pins LLVM and StableHLO to specific commits matching jaxlib 0.9.0 for bytecode compatibility (see the section on [Version Pinning](#version-pinning)) for details.

```bash
$ brew install cmake ninja
$ ./scripts/setup_deps.sh
```

2. Build the plugin and install it as a Python package. This step should be fast, and MUST be repeated for all changes to C++ files.

```bash
$ uv pip install -e .
```

### Version Pinning

The script pins LLVM and StableHLO to specific commits matching jaxlib 0.9.0 for bytecode compatibility. To update these versions for a different jaxlib release, trace the dependency chain:

```bash
# 1. Find XLA commit used by jaxlib
curl -s https://raw.githubusercontent.com/jax-ml/jax/jax-v0.9.0/third_party/xla/revision.bzl
# → XLA_COMMIT = "bb760b04..."

# 2. Find LLVM commit used by that XLA version
curl -s https://raw.githubusercontent.com/openxla/xla/<XLA_COMMIT>/third_party/llvm/workspace.bzl
# → LLVM_COMMIT = "f6d0a512..."

# 3. Find StableHLO commit used by that XLA version
curl -s https://raw.githubusercontent.com/openxla/xla/<XLA_COMMIT>/third_party/stablehlo/workspace.bzl
# → STABLEHLO_COMMIT = "127d2f23..."
```

Then update the `STABLEHLO_COMMIT` and `LLVM_COMMIT_OVERRIDE` variables in `setup_deps.sh`.

## Project Structure

```
jax-mps/
├── CMakeLists.txt
├── src/
│   ├── jax_plugins/mps/         # Python JAX plugin
│   ├── pjrt_plugin/             # C++ PJRT implementation
│   │   ├── pjrt_api.cc          # PJRT C API entry point
│   │   ├── mps_client.h/mm      # Metal client management
│   │   ├── mlx_executable.h/mm  # StableHLO compilation & MLX execution
│   │   └── ops/                 # Operation registry
│   └── proto/                   # Protobuf definitions
└── tests/
```

## How It Works

### PJRT Plugin

PJRT (Portable JAX Runtime) is JAX's abstraction for hardware backends. The plugin implements:

- `PJRT_Client_Create` - Initialize Metal device
- `PJRT_Client_Compile` - Parse StableHLO and build MLX operation graph
- `PJRT_Client_BufferFromHostBuffer` - Transfer data to GPU
- `PJRT_LoadedExecutable_Execute` - Run computation on GPU

### MLX Execution

StableHLO operations are mapped to MLX equivalents, e.g.:

- `stablehlo.add` → `mlx::core::add()`
- `stablehlo.dot_general` → `mlx::core::matmul()`
- `stablehlo.convolution` → `mlx::core::conv_general()`
- `stablehlo.reduce` → `mlx::core::sum/max/min/prod()`

[^1]: Measured against JAX's upstream test suite. Tests requiring float64 are excluded (MLX only supports float32). Tests requiring multiple devices or sharding are skipped automatically (single MPS device). Run with `uv run python scripts/run_jax_tests.py`.
