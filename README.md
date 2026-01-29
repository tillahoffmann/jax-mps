# jax-mps [![GitHub Action Badge](https://github.com/tillahoffmann/jax-mps/actions/workflows/build.yml/badge.svg)](https://github.com/tillahoffmann/jax-mps/actions/workflows/build.yml) [![PyPI](https://img.shields.io/pypi/v/jax-mps)](https://pypi.org/project/jax-mps/)

A JAX backend for Apple Metal Performance Shaders (MPS), enabling GPU-accelerated JAX computations on Apple Silicon.

## Example

jax-mps achieves a modest 3x speed-up over the CPU backend when training a simple ResNet18 model on CIFAR-10 using an M4 MacBook Air.

```bash
$ JAX_PLATFORMS=cpu uv run examples/resnet/main.py --steps=30
loss = 0.029: 100%|██████████| 30/30 [01:29<00:00,  2.99s/it]
Final training loss: 0.029
Time per step (second half): 3.041

$ JAX_PLATFORMS=mps uv run examples/resnet/main.py --steps=30
WARNING:2026-01-26 17:32:53,989:jax._src.xla_bridge:905: Platform 'mps' is experimental and not all JAX functionality may be correctly supported!
loss = 0.028: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Final training loss: 0.028
Time per step (second half): 0.991
```

## Installation

jax-mps requires macOS on Apple Silicon and Python 3.13. Install it with pip:

```bash
pip install jax-mps
```

The plugin registers itself with JAX automatically and is enabled by default. Set `JAX_PLATFORMS=mps` to select the MPS backend explicitly.

jax-mps is built against the StableHLO bytecode format matching jaxlib 0.9.x. Using a different jaxlib version will likely cause deserialization failures at JIT compile time. See [Version Pinning](#version-pinning) for details.

## Architecture

This project implements a [PJRT plugin](https://openxla.org/xla/pjrt) to offload evaluation of JAX expressions to a [Metal Performance Shaders Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph). The evaluation proceeds in several stages:

1. The JAX program is lowered to [StableHLO](https://openxla.org/stablehlo), a set of high-level operations for machine learning applications.
2. The plugin parses the StableHLO representation of the program and builds the corresponding MPS graph. The graph is cached to avoid re-construction on invocation of the same program, e.g., repeated training steps.
3. The MPS graph is executed, using native [MPS operations](./mps_ops/) where possible, and the results are returned to the caller.

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
│   │   ├── mps_executable.h/mm  # StableHLO compilation & execution
│   │   └── ops/                 # Operation implementations
│   └── proto/                   # Protobuf definitions
└── tests/
```

## How It Works

### PJRT Plugin

PJRT (Portable JAX Runtime) is JAX's abstraction for hardware backends. The plugin implements:

- `PJRT_Client_Create` - Initialize Metal device
- `PJRT_Client_Compile` - Parse HLO and prepare MPSGraph
- `PJRT_Client_BufferFromHostBuffer` - Transfer data to GPU
- `PJRT_LoadedExecutable_Execute` - Run computation on GPU

### MPSGraph Execution

Operations are mapped to MPSGraph equivalents, e.g.,:

- `add` → `additionWithPrimaryTensor:secondaryTensor:`
- `dot` → `matrixMultiplicationWithPrimaryTensor:secondaryTensor:`
- `tanh` → `tanhWithTensor:`
