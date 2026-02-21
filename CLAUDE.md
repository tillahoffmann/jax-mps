# Guidelines

- You may NEVER skip or xfail tests without my explicit approval.
- You MUST use `uv` to manage dependencies.
- You MUST use `uv run ...` to execute commands.
- You MUST check the comprehensive list of MPS Graph operations in `mps_ops/` before implementing a custom operation.
- You may NEVER use `--no-verify` for git commits.
- You may NEVER push to `main` unless explicitly requested.
- You may NEVER delete operations or tests without my explicit approval.
- For each op, you MUST register an `OperationTestConfig` for tests in `tests/test_ops.py`. See `tests/configs/unary.py` for an example and `tests/configs/util.py` for the signature of `OperationTestConfig`.
- You may NEVER create new op registries. Use `OpRegistry` for all ops and `CustomCallRegistry` for custom call targets. See `src/pjrt_plugin/ops/registry.h`.

# CI is Always Green

Tests, linting, and compilation on the `main` branch and in Continuous Integration testing ALWAYS pass. This means that any failures MUST be related to changes we made. You may NEVER claim that failures are "known issues," "unrelated to our changes," or similar.

# Naming Conventions

- Handler functions MUST use PascalCase: `HandleSort`, `HandleTopK`, `HandleDotGeneral`
- This follows Objective-C conventions for C functions in `.mm` files
- Macro-generated handlers also use PascalCase: `HandleMlir##suffix`, `HandleCc##suffix`

# Adding New Ops

1. Identify an op to implement and find its StableHLO op name (e.g., `stablehlo.cosine`). The simplest approach is to implement a test for the op and look for failures (the error message includes the StableHLO op name).
2. Find the matching MPS Graph method in `mps_ops/` (e.g., `cosWithTensor:name:` in `mps_ops/arithmetic.txt`).
3. For simple unary ops, add a `REGISTER_MLIR_UNARY_OP` line in `src/pjrt_plugin/ops/unary_ops.mm`:
   `REGISTER_MLIR_UNARY_OP("stablehlo.<name>", <mpsMethod>, <Suffix>);`
   where `<mpsMethod>` is the method prefix before `WithTensor:name:` and `<Suffix>` is PascalCase.
4. For binary/complex ops, write a handler function (e.g., `HandleMyOp`) and use `REGISTER_MPS_OP` (see existing examples in `src/pjrt_plugin/ops/`).
5. Rebuild with `uv pip install -e .` and run `uv run pytest` to confirm the XFAILs become PASSes.

# C++ Formatting

All C++ source files (`*.cc`, `*.mm`, `*.h`) MUST pass `clang-format`. Before committing, run:

```bash
find src -type f \( -name "*.cc" -o -name "*.mm" -o -name "*.h" \) | xargs clang-format -i
```

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
uv run pytest -m benchmark --benchmark-only
```
