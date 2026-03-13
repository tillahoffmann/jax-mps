# Guidelines

- You may NEVER skip or xfail tests without my explicit approval.
- You MUST use `uv` to manage dependencies.
- You MUST use `uv run ...` to execute commands.
- You may NEVER use `--no-verify` for git commits.
- You may NEVER push to `main` unless explicitly requested.
- You may NEVER delete operations or tests without my explicit approval.
- For each op, you MUST register an `OperationTestConfig` for tests in `tests/test_ops.py`. See `tests/configs/unary.py` for an example and `tests/configs/util.py` for the signature of `OperationTestConfig`.
- You may NEVER create new op registries. Use `OpRegistry` for all ops. See `src/pjrt_plugin/ops/registry.h`.

# CI is Always Green

Tests, linting, and compilation on the `main` branch and in Continuous Integration testing ALWAYS pass. This means that any failures MUST be related to changes we made. You may NEVER claim that failures are "known issues," "unrelated to our changes," or similar.

# Naming Conventions

- Handler functions MUST use PascalCase: `HandleAdd`, `HandleExp`, `HandleBroadcastInDim`

# Adding New Ops

1. Identify the StableHLO op name (e.g., `stablehlo.multiply`). Run a test and look for the error message which includes the op name.

2. Find the matching MLX function in the [MLX documentation](https://ml-explore.github.io/mlx/build/html/python/ops.html) (e.g., `mlx::core::multiply`).

3. Add a handler function in `src/pjrt_plugin/mlx_executable.mm`:
   ```cpp
   bool HandleMultiply(mlir::Operation* op, ValueMap& values, std::vector<mlx::core::array>& outputs, ExecContext& ctx) {
       auto lhs_opt = GetValue(values, op->getOperand(0));
       auto rhs_opt = GetValue(values, op->getOperand(1));
       if (!lhs_opt || !rhs_opt) {
           MPS_LOG_ERROR("stablehlo.multiply: operand not found\\n");
           return false;
       }
       values.emplace(ToKey(op->getResult(0)), mlx::core::multiply(lhs_opt->get(), rhs_opt->get()));
       return true;
   }
   ```

4. Register the handler in `GetOpHandlers()` in `src/pjrt_plugin/mlx_executable.mm`:
   ```cpp
   {"stablehlo.multiply", HandleMultiply},
   ```
   Op names are auto-derived from `GetOpHandlers()`, so no separate registration is needed.

5. Rebuild with `uv pip install -e .` and run tests.

# Build and Test

```bash
uv sync --all-groups
uv pip install -e .
uv run pytest
```

# Benchmarks

Benchmarks are excluded from normal test runs. To run them:

```bash
# Run benchmarks (compares CPU vs MPS performance)
uv run pytest -m benchmark --benchmark-only -n0
```
