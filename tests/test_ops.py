"""Tests for MPS operations comparing CPU vs GPU results.

Each operation is tested individually using pytest.mark.parametrize.
Tests verify that:
1. Results match CPU reference within tolerance
2. Operations actually run on the MPS device (not CPU fallback)
"""

import jax
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
