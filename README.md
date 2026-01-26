# jax-mps

A JAX backend for Apple Metal Performance Shaders (MPS), enabling GPU-accelerated JAX computations on Apple Silicon.

## Architecture

```
JAX Program
    ↓
StableHLO (MLIR)
    ↓
PJRT Plugin (this project)
    ↓
MPSGraph (Apple's graph framework)
    ↓
Metal GPU
```

This project implements a PJRT (Portable JAX Runtime) plugin that:
1. Receives HLO (High Level Operations) from JAX
2. Parses the HLO to identify operations
3. Builds an equivalent MPSGraph
4. Executes on the Metal GPU

## Requirements

- macOS 13.0 or later
- Apple Silicon (M1/M2/M3/M4) or AMD GPU
- CMake 3.20+, Ninja
- Xcode Command Line Tools
- Python 3.10+
- JAX 0.4.20+

## Building

```bash
# Install build tools
brew install cmake ninja

# Build and install LLVM/MLIR + StableHLO (one-time setup)
./scripts/setup_deps.sh

# Build jax-mps
cmake -B build -DCMAKE_PREFIX_PATH=$HOME/.local/jax-mps-deps
cmake --build build
```

The `setup_deps.sh` script:
- Clones LLVM and StableHLO
- Builds them against each other (they require matched versions)
- Installs to `~/.local/jax-mps-deps/` by default

Options:
```bash
./scripts/setup_deps.sh --prefix /custom/path  # Custom install location
./scripts/setup_deps.sh --jobs 4               # Limit parallel jobs
./scripts/setup_deps.sh --force                # Force rebuild with pinned versions
```

This will produce `build/lib/libpjrt_plugin_mps.dylib`.

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

## Installation

```bash
# Install the Python package in development mode
pip install -e .

# Or set the library path manually
export JAX_MPS_LIBRARY_PATH=/path/to/build/lib/libpjrt_plugin_mps.dylib
```

## Usage

```python
import jax
import jax.numpy as jnp

# The plugin registers automatically via entry points
# Or initialize manually:
from jax_plugins import mps
mps.initialize()

# Check available devices
print(jax.devices())  # Should show MPS device

# Use JAX as normal
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
print(x + y)
```

## Supported Operations

Currently implemented:
- **Binary ops**: `add`, `subtract`, `multiply`, `divide`, `maximum`, `minimum`, `remainder`, `power`
- **Matrix ops**: `dot`, `dot_general` (matrix multiplication), `convolution`
- **Unary ops**: `tanh`, `exp`, `log`, `log_plus_one`, `negate`, `abs`, `sqrt`, `rsqrt`, `erf`, `floor`, `sign`, `is_finite`
- **Comparison/selection**: `compare`, `select`, `clamp`
- **Shape ops**: `broadcast`, `broadcast_in_dim`, `reshape`, `transpose`, `convert`, `bitcast_convert`, `reverse`
- **Slicing/indexing**: `slice`, `dynamic_slice`, `dynamic_update_slice`, `gather`, `scatter`, `pad`, `iota`
- **Reduction ops**: `reduce` (sum, product, max, min, and, or)
- **Bitwise ops**: `and`, `or`, `xor`, `shift_left`, `shift_right_logical`
- **Other**: `concatenate`, `constant`, `custom_call`

Adding new operations: see `src/pjrt_plugin/ops/` for examples.

## Project Structure

```
jax-mps/
├── CMakeLists.txt              # Build configuration
├── src/pjrt_plugin/
│   ├── pjrt_api.cc             # PJRT C API entry point
│   ├── mps_client.h/mm         # Metal client management
│   ├── mps_device.h/mm         # Device abstraction
│   ├── mps_buffer.h/mm         # Buffer management
│   ├── mps_executable.h/mm     # StableHLO compilation & execution
│   ├── stablehlo_parser.h/mm   # MLIR-based StableHLO parsing
│   └── ops/                    # Operation implementations
│       ├── binary_ops.mm       # Arithmetic operations
│       ├── unary_ops.mm        # Math functions
│       ├── shape_ops.mm        # Reshape, transpose, etc.
│       ├── convolution_ops.mm  # Convolution operations
│       ├── reduction_ops.mm    # Reduction operations
│       └── bitwise_ops.mm      # Bitwise operations
├── python/
│   └── jax_plugins/
│       └── mps/
│           └── __init__.py     # JAX plugin registration
└── tests/
    ├── test_ops.py             # Operation tests
    ├── test_flax.py            # Flax integration tests
    └── test_rng.py             # RNG tests
```

## How It Works

### PJRT Plugin

PJRT (Portable JAX Runtime) is JAX's abstraction for hardware backends. Our plugin implements:
- `PJRT_Client_Create` - Initialize Metal device
- `PJRT_Client_Compile` - Parse HLO and prepare MPSGraph
- `PJRT_Client_BufferFromHostBuffer` - Transfer data to GPU
- `PJRT_LoadedExecutable_Execute` - Run computation on GPU

### HLO Parsing

We use MLIR and StableHLO libraries to parse the portable StableHLO bytecode format that JAX emits.

### MPSGraph Execution

Operations are mapped to MPSGraph equivalents:
- `add` → `additionWithPrimaryTensor:secondaryTensor:`
- `dot` → `matrixMultiplicationWithPrimaryTensor:secondaryTensor:`
- `tanh` → `tanhWithTensor:`

## Limitations

- **Subset of operations**: Many ops implemented, more can be added
- **Synchronous execution**: No async support yet
- **Single device**: Multi-GPU not supported

## Performance

On Apple M4, large matrix multiplication shows ~45-60x speedup over CPU:

| Operation | Size | CPU | MPS | Speedup |
|-----------|------|-----|-----|---------|
| matmul | 4000×4000 | 880ms | 15ms | **60x** |
| matmul | 1000×1000 | 8ms | 1.3ms | **6x** |
| add | 4000×4000 | 8ms | 13ms | 0.6x |

Small operations have GPU overhead that exceeds the computation benefit.

## Contributing

This is an experimental project. Contributions welcome:
1. Add more operations in `mps_executable.mm`
2. Add proper error handling
3. Implement async execution
4. Add tests

## References

- [OpenXLA PJRT Documentation](https://openxla.org/xla/pjrt)
- [PJRT Plugin Integration Guide](https://openxla.org/xla/pjrt/pjrt_integration)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [JAX Documentation](https://jax.readthedocs.io/)

## License

Apache License 2.0
