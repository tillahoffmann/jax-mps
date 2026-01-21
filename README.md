# jax-mps

A JAX backend for Apple Metal Performance Shaders (MPS), enabling GPU-accelerated JAX computations on Apple Silicon.

> **Status**: Early development / proof of concept

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

### Option 1: Lightweight Build (Quick Start)

Uses a hand-rolled StableHLO parser. Good for development and testing.

```bash
cmake -B build -DJAX_MPS_USE_MLIR=OFF
cmake --build build
```

### Option 2: Full Build with MLIR (Recommended)

Uses proper MLIR/StableHLO libraries for robust parsing. Requires building dependencies first.

```bash
# Install build tools
brew install cmake ninja

# Build and install LLVM/MLIR + StableHLO (one-time setup, ~30 min)
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
```

This will produce `build/lib/libpjrt_plugin_mps.dylib`.

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
- **Binary ops**: `add`, `subtract`, `multiply`, `divide`
- **Matrix ops**: `dot`, `dot_general` (matrix multiplication)
- **Unary ops**: `tanh`, `exp`, `log`, `negate`, `abs`
- **Shape ops**: `broadcast_in_dim`, `reshape`, `convert`

Adding new operations is straightforward - just add an entry to the `kOpHandlers` dispatch table in `src/pjrt_plugin/mps_executable.mm`.

## Project Structure

```
jax-mps/
├── CMakeLists.txt              # Build configuration
├── src/
│   ├── pjrt_plugin/
│   │   ├── pjrt_api.cc         # PJRT C API entry point
│   │   ├── mps_client.h/mm     # Metal client management
│   │   ├── mps_device.h/mm     # Device abstraction
│   │   ├── mps_buffer.h/mm     # Buffer management
│   │   └── mps_executable.h/mm # HLO parsing & execution
│   └── mps_backend/
│       └── mps_ops.mm          # MPSGraph operation helpers
├── python/
│   └── jax_plugins/
│       └── mps/
│           └── __init__.py     # JAX plugin registration
└── tests/
    └── test_basic.py
```

## How It Works

### PJRT Plugin

PJRT (Portable JAX Runtime) is JAX's abstraction for hardware backends. Our plugin implements:
- `PJRT_Client_Create` - Initialize Metal device
- `PJRT_Client_Compile` - Parse HLO and prepare MPSGraph
- `PJRT_Client_BufferFromHostBuffer` - Transfer data to GPU
- `PJRT_LoadedExecutable_Execute` - Run computation on GPU

### HLO Parsing

We implement a simple text-based HLO parser that recognizes patterns like:
```
%add = f32[2,3] add(%p0, %p1)
```

For production use, this should be replaced with proper protobuf-based HLO parsing.

### MPSGraph Execution

Operations are mapped to MPSGraph equivalents:
- `add` → `additionWithPrimaryTensor:secondaryTensor:`
- `dot` → `matrixMultiplicationWithPrimaryTensor:secondaryTensor:`
- `tanh` → `tanhWithTensor:`

## Limitations

- **Subset of operations**: Core ops implemented, more can be added
- **No autodiff**: Gradients not yet supported through this backend
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
2. Improve HLO parsing
3. Add proper error handling
4. Implement async execution
5. Add tests

## References

- [OpenXLA PJRT Documentation](https://openxla.org/xla/pjrt)
- [PJRT Plugin Integration Guide](https://openxla.org/xla/pjrt/pjrt_integration)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [JAX Documentation](https://jax.readthedocs.io/)

## License

Apache License 2.0
