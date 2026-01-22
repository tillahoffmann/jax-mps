"""Tests for MPS operations comparing CPU vs GPU results.

Each operation is tested individually using pytest.mark.parametrize.
Tests verify that:
1. Results match CPU reference within tolerance
2. Operations actually run on the MPS device (not CPU fallback)
"""

import numpy as np
import pytest
from conftest import assert_on_mps
from numpy.testing import assert_allclose

# Binary operations
BINARY_OPS = [
    pytest.param("add", lambda jnp, a, b: jnp.add(a, b), id="add"),
    pytest.param("subtract", lambda jnp, a, b: jnp.subtract(a, b), id="subtract"),
    pytest.param("multiply", lambda jnp, a, b: jnp.multiply(a, b), id="multiply"),
    pytest.param("divide", lambda jnp, a, b: jnp.divide(a, b), id="divide"),
]


@pytest.mark.parametrize("name,op_fn", BINARY_OPS)
def test_binary_op(jax_setup, name, op_fn):
    """Test binary operations produce correct results on MPS."""
    jax = jax_setup["jax"]
    jnp = jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    # Test data - avoid zeros for divide
    np.random.seed(42)
    a_np = np.random.randn(32, 32).astype(np.float32)
    b_np = np.random.randn(32, 32).astype(np.float32) + 0.1

    # Run on CPU for reference
    a_cpu = jax.device_put(a_np, cpu)
    b_cpu = jax.device_put(b_np, cpu)
    expected = np.array(op_fn(jnp, a_cpu, b_cpu))

    # Run on MPS
    a_mps = jax.device_put(a_np, mps)
    b_mps = jax.device_put(b_np, mps)
    result = op_fn(jnp, a_mps, b_mps)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness
    assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)


# Matrix multiplication (dot)
@pytest.mark.parametrize(
    "shape_a,shape_b",
    [
        pytest.param((32, 64), (64, 32), id="matmul_32x64_64x32"),
        pytest.param((16, 16), (16, 16), id="matmul_16x16_16x16"),
        pytest.param((64, 128), (128, 64), id="matmul_64x128_128x64"),
    ],
)
def test_dot(jax_setup, shape_a, shape_b):
    """Test matrix multiplication produces correct results on MPS."""
    jax = jax_setup["jax"]
    jnp = jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    a_np = np.random.randn(*shape_a).astype(np.float32)
    b_np = np.random.randn(*shape_b).astype(np.float32)

    # Run on CPU for reference
    a_cpu = jax.device_put(a_np, cpu)
    b_cpu = jax.device_put(b_np, cpu)
    expected = np.array(jnp.matmul(a_cpu, b_cpu))

    # Run on MPS
    a_mps = jax.device_put(a_np, mps)
    b_mps = jax.device_put(b_np, mps)
    result = jnp.matmul(a_mps, b_mps)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness (matmul can accumulate errors)
    assert_allclose(np.array(result), expected, rtol=1e-4, atol=1e-4)


# Maximum and Minimum operations
@pytest.mark.parametrize(
    "name,op_fn",
    [
        pytest.param("maximum", lambda jnp, a, b: jnp.maximum(a, b), id="maximum"),
        pytest.param("minimum", lambda jnp, a, b: jnp.minimum(a, b), id="minimum"),
    ],
)
def test_minmax_op(jax_setup, name, op_fn):
    """Test maximum and minimum operations produce correct results on MPS."""
    jax = jax_setup["jax"]
    jnp = jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    a_np = np.random.randn(32, 32).astype(np.float32)
    b_np = np.random.randn(32, 32).astype(np.float32)

    # Run on CPU for reference
    a_cpu = jax.device_put(a_np, cpu)
    b_cpu = jax.device_put(b_np, cpu)
    expected = np.array(op_fn(jnp, a_cpu, b_cpu))

    # Run on MPS
    a_mps = jax.device_put(a_np, mps)
    b_mps = jax.device_put(b_np, mps)
    result = op_fn(jnp, a_mps, b_mps)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness
    assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)


# ReLU activation (uses maximum internally)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32,), id="relu_1d"),
        pytest.param((16, 16), id="relu_2d"),
        pytest.param((4, 8, 8), id="relu_3d"),
    ],
)
def test_relu(jax_setup, shape):
    """Test ReLU activation produces correct results on MPS."""
    jax = jax_setup["jax"]
    jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    # Use data with both positive and negative values
    x_np = np.random.randn(*shape).astype(np.float32)

    # Run on CPU for reference
    x_cpu = jax.device_put(x_np, cpu)
    expected = np.array(jax.nn.relu(x_cpu))

    # Run on MPS
    x_mps = jax.device_put(x_np, mps)
    result = jax.nn.relu(x_mps)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness
    assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)


# Unary operations
UNARY_OPS = [
    pytest.param("tanh", lambda jnp, x: jnp.tanh(x), id="tanh"),
    pytest.param("exp", lambda jnp, x: jnp.exp(x), id="exp"),
    pytest.param("log", lambda jnp, x: jnp.log(x), id="log"),
    pytest.param("negate", lambda jnp, x: jnp.negative(x), id="negate"),
    pytest.param("abs", lambda jnp, x: jnp.abs(x), id="abs"),
]


@pytest.mark.parametrize("name,op_fn", UNARY_OPS)
def test_unary_op(jax_setup, name, op_fn):
    """Test unary operations produce correct results on MPS."""
    jax = jax_setup["jax"]
    jnp = jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    # Use positive values for log, reasonable range for exp
    if name == "log":
        x_np = np.abs(np.random.randn(32, 32).astype(np.float32)) + 0.1
    elif name == "exp":
        x_np = np.random.randn(32, 32).astype(np.float32) * 0.5  # Avoid overflow
    else:
        x_np = np.random.randn(32, 32).astype(np.float32)

    # Run on CPU for reference
    x_cpu = jax.device_put(x_np, cpu)
    expected = np.array(op_fn(jnp, x_cpu))

    # Run on MPS
    x_mps = jax.device_put(x_np, mps)
    result = op_fn(jnp, x_mps)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness
    assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)


# Shape operations
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        pytest.param((4, 8), (8, 4), id="reshape_4x8_to_8x4"),
        pytest.param((2, 3, 4), (6, 4), id="reshape_2x3x4_to_6x4"),
        pytest.param((32,), (4, 8), id="reshape_32_to_4x8"),
        pytest.param((2, 2, 2, 2), (4, 4), id="reshape_2x2x2x2_to_4x4"),
    ],
)
def test_reshape(jax_setup, input_shape, output_shape):
    """Test reshape produces correct results on MPS."""
    jax = jax_setup["jax"]
    jnp = jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    x_np = np.random.randn(*input_shape).astype(np.float32)

    # Run on CPU for reference
    x_cpu = jax.device_put(x_np, cpu)
    expected = np.array(jnp.reshape(x_cpu, output_shape))

    # Run on MPS
    x_mps = jax.device_put(x_np, mps)
    result = jnp.reshape(x_mps, output_shape)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness (reshape should be exact)
    assert_allclose(np.array(result), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        pytest.param((1,), (4,), id="broadcast_1_to_4"),
        pytest.param((1, 4), (3, 4), id="broadcast_1x4_to_3x4"),
        pytest.param((4, 1), (4, 8), id="broadcast_4x1_to_4x8"),
    ],
)
def test_broadcast(jax_setup, input_shape, output_shape):
    """Test broadcast produces correct results on MPS."""
    jax = jax_setup["jax"]
    jnp = jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    x_np = np.random.randn(*input_shape).astype(np.float32)

    # Run on CPU for reference
    x_cpu = jax.device_put(x_np, cpu)
    expected = np.array(jnp.broadcast_to(x_cpu, output_shape))

    # Run on MPS
    x_mps = jax.device_put(x_np, mps)
    result = jnp.broadcast_to(x_mps, output_shape)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness (broadcast should be exact)
    assert_allclose(np.array(result), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "from_dtype,to_dtype",
    [
        pytest.param(np.float32, np.float16, id="f32_to_f16"),
        pytest.param(np.float16, np.float32, id="f16_to_f32"),
        pytest.param(np.int32, np.float32, id="i32_to_f32"),
        pytest.param(np.float32, np.int32, id="f32_to_i32"),
    ],
)
def test_convert(jax_setup, from_dtype, to_dtype):
    """Test type conversion produces correct results on MPS."""
    jax = jax_setup["jax"]
    jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    if np.issubdtype(from_dtype, np.integer):
        x_np = np.random.randint(-100, 100, size=(16, 16)).astype(from_dtype)
    else:
        x_np = np.random.randn(16, 16).astype(from_dtype)

    # Run on CPU for reference
    x_cpu = jax.device_put(x_np, cpu)
    expected = np.array(x_cpu.astype(to_dtype))

    # Run on MPS
    x_mps = jax.device_put(x_np, mps)
    result = x_mps.astype(to_dtype)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness (allow some tolerance for float conversions)
    if np.issubdtype(to_dtype, np.floating):
        assert_allclose(np.array(result), expected, rtol=1e-3, atol=1e-3)
    else:
        assert_allclose(np.array(result), expected, rtol=0, atol=0)


# Composite operations (to test operation chaining)
@pytest.mark.parametrize(
    "name,op_fn",
    [
        pytest.param(
            "add_mul",
            lambda jnp, a, b, c: jnp.multiply(jnp.add(a, b), c),
            id="add_then_mul",
        ),
        pytest.param(
            "tanh_add",
            lambda jnp, a, b, c: jnp.add(jnp.tanh(a), jnp.tanh(b)),
            id="tanh_then_add",
        ),
        pytest.param(
            "matmul_tanh",
            lambda jnp, a, b, c: jnp.tanh(jnp.matmul(a, b)),
            id="matmul_then_tanh",
        ),
    ],
)
def test_composite_op(jax_setup, name, op_fn):
    """Test composite operations produce correct results on MPS."""
    jax = jax_setup["jax"]
    jnp = jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    a_np = np.random.randn(16, 16).astype(np.float32)
    b_np = np.random.randn(16, 16).astype(np.float32)
    c_np = np.random.randn(16, 16).astype(np.float32)

    # Run on CPU for reference
    a_cpu = jax.device_put(a_np, cpu)
    b_cpu = jax.device_put(b_np, cpu)
    c_cpu = jax.device_put(c_np, cpu)
    expected = np.array(op_fn(jnp, a_cpu, b_cpu, c_cpu))

    # Run on MPS
    a_mps = jax.device_put(a_np, mps)
    b_mps = jax.device_put(b_np, mps)
    c_mps = jax.device_put(c_np, mps)
    result = op_fn(jnp, a_mps, b_mps, c_mps)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness
    assert_allclose(np.array(result), expected, rtol=1e-4, atol=1e-4)


# JIT compilation tests
@pytest.mark.parametrize("name,op_fn", BINARY_OPS[:2])  # Test a couple of ops
def test_jit_binary_op(jax_setup, name, op_fn):
    """Test JIT-compiled binary operations produce correct results on MPS."""
    jax = jax_setup["jax"]
    jnp = jax_setup["jnp"]
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    np.random.seed(42)
    a_np = np.random.randn(32, 32).astype(np.float32)
    b_np = np.random.randn(32, 32).astype(np.float32) + 0.1

    # Create JIT-compiled function
    @jax.jit
    def jit_fn(a, b):
        return op_fn(jnp, a, b)

    # Run on CPU for reference
    a_cpu = jax.device_put(a_np, cpu)
    b_cpu = jax.device_put(b_np, cpu)
    expected = np.array(jit_fn(a_cpu, b_cpu))

    # Run on MPS
    a_mps = jax.device_put(a_np, mps)
    b_mps = jax.device_put(b_np, mps)
    result = jit_fn(a_mps, b_mps)

    # Verify result is on MPS
    assert_on_mps(result, mps)

    # Verify correctness
    assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)
