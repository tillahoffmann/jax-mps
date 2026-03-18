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

Pre-commit hooks run clang-format, ruff, pyright, a rebuild, and the full test suite on every commit. MPS is not available in GitHub Actions, so the pre-commit hooks are the primary line of defence — please do not skip them. This may seem pedantic (apologies), but agents need strong guardrails in the form of validation so they don't ... go off the rails.

## Adding a new operation

1. **Find the MLX function matching the operation.** See the [MLX C++ documentation](https://ml-explore.github.io/mlx/build/html/python/ops.html) (Python API mirrors C++).

2. **Add a handler function** in `src/pjrt_plugin/mlx_executable.cc`:

```cpp
bool HandleCosine(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
    auto input_opt = GetValue(values, op->getOperand(0));
    if (!input_opt) {
        MPS_LOG_ERROR("stablehlo.cosine: operand not found\n");
        return false;
    }
    values.emplace(ToKey(op->getResult(0)), mlx::core::cos(input_opt->get()));
    return true;
}
```

3. **Register the handler** in `GetOpHandlers()`:

```cpp
{"stablehlo.cosine", HandleCosine},
```

Op names are auto-derived from `GetOpHandlers()`, so no separate registration is needed.

4. **Add a test config.** Every op needs an `OperationTestConfig` entry in the appropriate file under `tests/configs/`. See `tests/configs/unary.py` for the pattern.

5. **Rebuild and test.** C++ changes require a rebuild.

```bash
uv pip install -e .
uv run pytest
```

## Pull requests

Please open PRs against `main`. The pre-commit hooks ensure formatting, type checking, and tests all pass before a commit is created.
