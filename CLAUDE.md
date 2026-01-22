# Guidelines

- You may NEVER skip or xfail tests without my explicit approval.
- You MUST use `uv` to manage dependencies and for execution.
- You MUST check the comprehensive list of MPS Graph operations in `mps_ops/` before implementing a custom operation.
- You may NEVER use `--no-verify` for git commits.
- You may NEVER delete operations or tests without my explicit approval.
- For each op, you MUST add a test to `tests/test_ops.py`, including the `assert_cpu_mps_allclose` decorator for test functions.

# Build and Test

```bash
uv sync
uv pip install -e .
uv run pytest
```
