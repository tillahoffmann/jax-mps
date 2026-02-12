# Copilot Code Review Instructions

JAX PJRT plugin for Apple Metal Performance Shaders. Review PRs against these requirements.

## Review Philosophy

Prioritize high-impact feedback. Focus review effort on:

- Bugs, correctness issues, and security vulnerabilities
- Architectural violations (see guidelines below)
- Missing tests for new functionality
- Performance problems with measurable impact

Avoid commenting on:

- Speculative future scenarios that aren't part of the current change
- Availability of standard Unix tools (bc, awk, grep) in developer scripts
- Constants whose meaning is clear from context
- Alternative implementations when the current code is correct and readable

## Architecture

- Reject ops that modify core files. New ops must be self-contained in `src/pjrt_plugin/ops/`.
- Core files (flag if touched by op PRs):
  - `mps_executable.{h,mm}`
  - `stablehlo_parser.{h,mm}`
  - `pjrt_*.{h,cc}`
  - `mps_client.{h,mm}`, `mps_device.{h,mm}`, `mps_buffer.{h,mm}`

## Naming Conventions

- Use PascalCase for handler functions (Objective-C convention).

```cpp
// Correct
ProcessResult HandleDotGeneral(...);
ProcessResult HandleTopK(...);

// Incorrect - flag these
ProcessResult handleDotGeneral(...);  // camelCase
ProcessResult handle_dot_general(...);  // snake_case
```

## Code Style

- Use existing registration macros. Do not create new registries.

```objc
// Correct - simple unary op
REGISTER_MLIR_UNARY_OP("stablehlo.tanh", tanh, Tanh);

// Correct - complex handler
REGISTER_MPS_OP("stablehlo.dot_general", HandleDotGeneral);

// Incorrect - new registry
static std::map<std::string, Handler> myCustomRegistry;  // Flag this
```

- Use `type_utils.{h,mm}` for type conversions. Do not hand-roll conversions.

```cpp
// Correct
MPSDataType mpsType = MlirTypeToMps(mlirType);

// Incorrect - flag manual conversion
MPSDataType mpsType;
if (mlirType.isF32()) mpsType = MPSDataTypeFloat32;  // Use type_utils instead
```

## Testing Guidelines

- Every op requires an `OperationTestConfig` in `tests/configs/`.
- Flag PRs adding ops without corresponding test configs.
- Flag any `@pytest.mark.skip` or `@pytest.mark.xfail` without maintainer approval.

```python
# Required pattern for each op
OperationTestConfig(
    func=jnp.tanh,
    args=[lambda rng: rng.standard_normal((3, 4))],
    seed=42,
)
```

- `differentiable_argnums=()` is only valid when no arguments are truly differentiable (e.g., integer/boolean ops). Flag uses that skip gradient tests for ops with float inputs.

```python
# Correct - bitwise ops on integers have no gradients
OperationTestConfig(func=jnp.bitwise_and, ..., differentiable_argnums=())

# Incorrect - tanh has differentiable float input, must test gradients
OperationTestConfig(func=jnp.tanh, ..., differentiable_argnums=())  # Flag this
```

- Test configs should exercise the op's behavior, not just prove it runs. Flag configs that only test a single shape or ignore the op's parameters. If an op takes an `axis` argument, test different axes. If it supports batching, test batched inputs. If it has configuration kwargs, vary them.

## Common Mistakes

- **Not using helpers**: Use `GetInputTensor()` and `Result()` instead of manual ValueMap manipulation.

```cpp
// Correct
MPSGraphTensor* input = GetInputTensor(values, op, 0);
return Result(values, op, output, "my_op");

// Incorrect - manual ValueMap access
auto* input = values[op->getOperand(0).getAsOpaquePointer()];  // Use GetInputTensor
values[op->getResult(0).getAsOpaquePointer()] = output;  // Use Result
```

## Review Checklist

- Changes confined to `src/pjrt_plugin/ops/` and `tests/configs/`
- Handler uses PascalCase
- Uses `REGISTER_*` macros, no new registries
- Uses `GetInputTensor()` and `Result()` helpers
- Test config present in `tests/configs/`
- No modifications to core infrastructure files
- No unexplained skip/xfail markers
- `differentiable_argnums=()` only for truly non-differentiable ops
- Test configs exercise the op's parameters and edge cases, not just a single happy path
