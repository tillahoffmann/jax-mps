# Guidelines

- You may NEVER skip or xfail tests without my explicit approval.
- You MUST use `uv` to manage dependencies and for execution.

# Build and Test

```bash
# Install dependencies
uv sync

# Build and install (editable mode)
uv pip install -e .

# Run tests
uv run pytest
```
