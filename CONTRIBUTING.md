# Contributing

This guide walks through the development setup and the workflow for adding new operations.

## Setup

You need macOS on Apple Silicon, Python 3.13, and [uv](https://docs.astral.sh/uv/). Start by building the LLVM/MLIR and StableHLO dependencies. This is a one-time step and takes about 30 minutes.

```bash
brew install cmake ninja
./scripts/setup_deps.sh
```

Then install the Python dependencies, build the plugin, and set up pre-commit hooks:

```bash
uv sync --all-groups
uv pip install -e .
pre-commit install
```

Pre-commit hooks run clang-format, ruff, pyright, a rebuild, and the full test suite on every commit. MPS is not available in GitHub Actions, so the pre-commit hooks are the primary line of defence — please do not skip them.

## Adding a new operation

1. **Find the MPS Graph method matching the operation you want to implement.** The `mps_ops/` directory contains a categorised list of all MPSGraph methods, extracted from the framework headers under `MPSGraph.framework/Headers/`. Apple's [MPSGraph documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph) is the authoritative reference.

2. **Register the op.** For simple unary ops, a one-liner in `src/pjrt_plugin/ops/unary_ops.mm` is enough (see [#12](https://github.com/tillahoffmann/jax-mps/pull/12) for an example):

```objc
REGISTER_MLIR_UNARY_OP("stablehlo.cosine", cos, cosine);
```

The second argument is the MPS method prefix (before `WithTensor:name:`), and the third is a unique suffix for the registration symbol. There is an analogous `REGISTER_MLIR_BINARY_OP` macro for binary ops. For anything more involved, write a handler function and use `REGISTER_MPS_OP` — see the existing handlers for examples.

3. **Add a test config.** Every op needs an `OperationTestConfig` entry in the appropriate file under `tests/configs/`. See `tests/configs/unary.py` for the pattern.

4. **Rebuild and test.** C++ changes require a rebuild.

```bash
uv pip install -e .
uv run pytest
```

## Pull requests

Please open PRs against `main`. The pre-commit hooks ensure formatting, type checking, and tests all pass before a commit is created.
