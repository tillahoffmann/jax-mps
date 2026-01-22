"""Tests for MPS operations comparing CPU vs GPU results.

Each operation is tested individually using pytest.mark.parametrize.
Tests verify that:
1. Results match CPU reference within tolerance
2. Operations actually run on the MPS device (not CPU fallback)
"""

import jax
import jax.scipy.special
import numpy as np
import pytest
from conftest import assert_cpu_mps_allclose
from jax import numpy as jnp


# Binary operations using new decorator pattern
@pytest.mark.parametrize("op", [jnp.add, jnp.subtract, jnp.multiply, jnp.divide])
@pytest.mark.parametrize(
    "a, b",
    [
        (np.random.normal(), np.random.normal()),
        (np.random.normal(size=(3,)), np.random.normal(size=(3,))),
        (np.random.normal(size=(4, 1)), np.random.normal(size=(3,))),
        (
            np.random.randn(32, 32).astype(np.float32),
            np.random.randn(32, 32).astype(np.float32) + 0.1,
        ),
    ],
)
@assert_cpu_mps_allclose
def test_binary_op(request: pytest.FixtureRequest, device, op, a, b):
    return op(a, b)


# Matrix multiplication
@pytest.mark.parametrize(
    "a, b",
    [
        (
            np.random.randn(32, 64).astype(np.float32),
            np.random.randn(64, 32).astype(np.float32),
        ),
        (
            np.random.randn(16, 16).astype(np.float32),
            np.random.randn(16, 16).astype(np.float32),
        ),
        (
            np.random.randn(64, 128).astype(np.float32),
            np.random.randn(128, 64).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_matmul(request: pytest.FixtureRequest, device, a, b):
    return jnp.matmul(a, b)


# Min/max operations
@pytest.mark.parametrize("op", [jnp.maximum, jnp.minimum])
@pytest.mark.parametrize(
    "a, b",
    [
        (
            np.random.randn(32, 32).astype(np.float32),
            np.random.randn(32, 32).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_minmax_op(request: pytest.FixtureRequest, device, op, a, b):
    return op(a, b)


# ReLU activation
@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(32).astype(np.float32),
        np.random.randn(16, 16).astype(np.float32),
        np.random.randn(4, 8, 8).astype(np.float32),
    ],
)
@assert_cpu_mps_allclose
def test_relu(request: pytest.FixtureRequest, device, x):
    return jax.nn.relu(x)


# Unary operations
@pytest.mark.parametrize(
    "op, x",
    [
        (jnp.tanh, np.random.randn(32, 32).astype(np.float32)),
        (jnp.exp, np.random.randn(32, 32).astype(np.float32) * 0.5),  # Avoid overflow
        (
            jnp.log,
            np.abs(np.random.randn(32, 32).astype(np.float32)) + 0.1,
        ),  # Positive values
        (jnp.negative, np.random.randn(32, 32).astype(np.float32)),
        (jnp.abs, np.random.randn(32, 32).astype(np.float32)),
    ],
)
@assert_cpu_mps_allclose
def test_unary_op(request: pytest.FixtureRequest, device, op, x):
    return op(x)


# Reshape operations
@pytest.mark.parametrize(
    "x, output_shape",
    [
        (np.random.randn(4, 8).astype(np.float32), (8, 4)),
        (np.random.randn(2, 3, 4).astype(np.float32), (6, 4)),
        (np.random.randn(32).astype(np.float32), (4, 8)),
        (np.random.randn(2, 2, 2, 2).astype(np.float32), (4, 4)),
    ],
)
@assert_cpu_mps_allclose
def test_reshape(request: pytest.FixtureRequest, device, x, output_shape):
    return jnp.reshape(x, output_shape)


# Broadcast operations
@pytest.mark.parametrize(
    "x, output_shape",
    [
        (np.random.randn(1).astype(np.float32), (4,)),
        (np.random.randn(1, 4).astype(np.float32), (3, 4)),
        (np.random.randn(4, 1).astype(np.float32), (4, 8)),
    ],
)
@assert_cpu_mps_allclose
def test_broadcast(request: pytest.FixtureRequest, device, x, output_shape):
    return jnp.broadcast_to(x, output_shape)


# Type conversion
@pytest.mark.parametrize(
    "x, to_dtype",
    [
        (np.random.randn(16, 16).astype(np.float32), np.float16),
        (np.random.randn(16, 16).astype(np.float16), np.float32),
        (np.random.randint(-100, 100, size=(16, 16)).astype(np.int32), np.float32),
        (np.random.randn(16, 16).astype(np.float32), np.int32),
    ],
)
@assert_cpu_mps_allclose
def test_convert(request: pytest.FixtureRequest, device, x, to_dtype):
    return x.astype(to_dtype)


# Additional unary operations
@pytest.mark.parametrize(
    "op, x",
    [
        (jnp.sqrt, np.abs(np.random.randn(32, 32).astype(np.float32)) + 0.1),
        (jnp.log1p, np.abs(np.random.randn(32, 32).astype(np.float32))),
        (jax.scipy.special.erf, np.random.randn(32, 32).astype(np.float32) * 0.5),
        (
            jax.scipy.special.erfinv,
            np.random.uniform(-0.9, 0.9, (32, 32)).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_special_unary_op(request: pytest.FixtureRequest, device, op, x):
    return op(x)


# Clip/clamp operation
@pytest.mark.parametrize(
    "x, a_min, a_max",
    [
        (np.random.randn(32, 32).astype(np.float32), -0.5, 0.5),
        (np.random.randn(16, 16).astype(np.float32), 0.0, 1.0),
    ],
)
@assert_cpu_mps_allclose
def test_clip(request: pytest.FixtureRequest, device, x, a_min, a_max):
    return jnp.clip(x, a_min, a_max)


# Transpose operation
@pytest.mark.parametrize(
    "x, axes",
    [
        (np.random.randn(4, 8).astype(np.float32), (1, 0)),
        (np.random.randn(2, 3, 4).astype(np.float32), (2, 0, 1)),
        (np.random.randn(2, 3, 4, 5).astype(np.float32), (3, 2, 1, 0)),
    ],
)
@assert_cpu_mps_allclose
def test_transpose(request: pytest.FixtureRequest, device, x, axes):
    return jnp.transpose(x, axes)


# Slice operation
@pytest.mark.parametrize(
    "shape, slices",
    [
        ((10,), (slice(2, 8),)),
        ((8, 8), (slice(1, 5), slice(2, 6))),
        ((4, 8, 8), (slice(1, 3), slice(2, 6), slice(0, 4))),
    ],
)
@assert_cpu_mps_allclose
def test_slice(request: pytest.FixtureRequest, device, shape, slices):
    rng = np.random.default_rng(seed=42)
    x = jax.device_put(rng.standard_normal(shape).astype(np.float32), device)

    @jax.jit
    def do_slice(x):
        return x[slices]

    return do_slice(x)


# Concatenate operation
@pytest.mark.parametrize(
    "arrays, axis",
    [
        ([np.random.randn(4, 8).astype(np.float32) for _ in range(3)], 0),
        ([np.random.randn(4, 8).astype(np.float32) for _ in range(2)], 1),
        ([np.random.randn(2, 3, 4).astype(np.float32) for _ in range(2)], 2),
    ],
)
@assert_cpu_mps_allclose
def test_concatenate(request: pytest.FixtureRequest, device, arrays, axis):
    return jnp.concatenate(arrays, axis=axis)


# Iota/arange operation
@pytest.mark.parametrize(
    "start, stop, dtype",
    [
        (0, 10, np.float32),
        (0, 32, np.int32),
        (5, 15, np.float32),
    ],
)
@assert_cpu_mps_allclose
def test_arange(request: pytest.FixtureRequest, device, start, stop, dtype):
    return jnp.arange(start, stop, dtype=dtype)


# Bitwise operations
@pytest.mark.parametrize(
    "op",
    [jnp.bitwise_and, jnp.bitwise_or, jnp.bitwise_xor],
)
@pytest.mark.parametrize(
    "a, b",
    [
        (
            np.random.randint(0, 256, size=(32, 32)).astype(np.uint32),
            np.random.randint(0, 256, size=(32, 32)).astype(np.uint32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_bitwise_op(request: pytest.FixtureRequest, device, op, a, b):
    return op(a, b)


# Shift operations
@pytest.mark.parametrize(
    "op",
    [jnp.left_shift, jnp.right_shift],
)
@pytest.mark.parametrize(
    "a, b",
    [
        (
            np.random.randint(0, 256, size=(32, 32)).astype(np.uint32),
            np.random.randint(0, 8, size=(32, 32)).astype(np.uint32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_shift_op(request: pytest.FixtureRequest, device, op, a, b):
    return op(a, b)


# Composite operations (operation chaining)
@pytest.mark.parametrize(
    "op_fn",
    [
        lambda a, b, c: jnp.multiply(jnp.add(a, b), c),
        lambda a, b, c: jnp.add(jnp.tanh(a), jnp.tanh(b)),
        lambda a, b, c: jnp.tanh(jnp.matmul(a, b)),
    ],
)
@pytest.mark.parametrize(
    "a, b, c",
    [
        (
            np.random.randn(16, 16).astype(np.float32),
            np.random.randn(16, 16).astype(np.float32),
            np.random.randn(16, 16).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_composite_op(request: pytest.FixtureRequest, device, op_fn, a, b, c):
    return op_fn(a, b, c)


# JIT compilation tests
@pytest.mark.parametrize("op", [jnp.add, jnp.subtract])
@pytest.mark.parametrize(
    "a, b",
    [
        (
            np.random.randn(32, 32).astype(np.float32),
            np.random.randn(32, 32).astype(np.float32) + 0.1,
        ),
    ],
)
@assert_cpu_mps_allclose
def test_jit_binary_op(request: pytest.FixtureRequest, device, op, a, b):
    @jax.jit
    def jit_fn(a, b):
        return op(a, b)

    return jit_fn(a, b)
