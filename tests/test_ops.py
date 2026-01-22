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
from conftest import assert_cpu_mps_allclose, register_op_test
from jax import numpy as jnp

# Register ops that are implicitly tested by every test or used internally
register_op_test(
    "func.return", "func.call", "stablehlo.constant", "stablehlo.custom_call"
)

# Common test data
_rng = np.random.default_rng(42)
_float_2d = _rng.standard_normal((32, 32)).astype(np.float32)
_float_2d_b = _rng.standard_normal((32, 32)).astype(np.float32) + 0.1
_float_positive = np.abs(_rng.standard_normal((32, 32)).astype(np.float32)) + 0.1
_uint_2d = _rng.integers(0, 256, size=(32, 32)).astype(np.uint32)
_uint_2d_b = _rng.integers(0, 256, size=(32, 32)).astype(np.uint32)
_uint_shift = _rng.integers(0, 8, size=(32, 32)).astype(np.uint32)


# Unary operations
@pytest.mark.parametrize(
    "op, x",
    [
        (register_op_test(jnp.tanh, "stablehlo.tanh"), _float_2d),
        (register_op_test(jnp.exp, "stablehlo.exponential"), _float_2d * 0.5),
        (register_op_test(jnp.log, "stablehlo.log"), _float_positive),
        (register_op_test(jnp.negative, "stablehlo.negate"), _float_2d),
        (register_op_test(jnp.abs, "stablehlo.abs"), _float_2d),
        (register_op_test(jnp.sqrt, "stablehlo.sqrt"), _float_positive),
        (register_op_test(jnp.log1p, "stablehlo.log_plus_one"), _float_positive),
        (register_op_test(jax.scipy.special.erf, "stablehlo.erf"), _float_2d * 0.5),
        (
            register_op_test(jax.scipy.special.erfinv, "chlo.erf_inv"),
            _rng.uniform(-0.9, 0.9, (32, 32)).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_unary_op(request: pytest.FixtureRequest, device, op, x):
    return op(x)


# Binary operations (arithmetic, min/max, remainder, bitwise, shifts)
@pytest.mark.parametrize(
    "op, a, b",
    [
        # Arithmetic
        (register_op_test(jnp.add, "stablehlo.add"), _float_2d, _float_2d_b),
        (register_op_test(jnp.subtract, "stablehlo.subtract"), _float_2d, _float_2d_b),
        (register_op_test(jnp.multiply, "stablehlo.multiply"), _float_2d, _float_2d_b),
        (register_op_test(jnp.divide, "stablehlo.divide"), _float_2d, _float_2d_b),
        (register_op_test(jnp.maximum, "stablehlo.maximum"), _float_2d, _float_2d_b),
        (register_op_test(jnp.minimum, "stablehlo.minimum"), _float_2d, _float_2d_b),
        (
            register_op_test(jnp.remainder, "stablehlo.remainder"),
            _float_2d * 10,
            _float_2d_b * 3 + 1,
        ),
        (
            register_op_test(jnp.nextafter, "chlo.next_after"),
            np.array([1.0, -1.0, 0.0, 2.0], dtype=np.float32),
            np.array([2.0, -2.0, 1.0, 1.0], dtype=np.float32),
        ),
        # Bitwise
        (register_op_test(jnp.bitwise_and, "stablehlo.and"), _uint_2d, _uint_2d_b),
        (register_op_test(jnp.bitwise_or, "stablehlo.or"), _uint_2d, _uint_2d_b),
        (register_op_test(jnp.bitwise_xor, "stablehlo.xor"), _uint_2d, _uint_2d_b),
        # Shifts
        (
            register_op_test(jnp.left_shift, "stablehlo.shift_left"),
            _uint_2d,
            _uint_shift,
        ),
        (
            register_op_test(jnp.right_shift, "stablehlo.shift_right_logical"),
            _uint_2d,
            _uint_shift,
        ),
    ],
)
@assert_cpu_mps_allclose
def test_binary_op(request: pytest.FixtureRequest, device, op, a, b):
    return op(a, b)


# Matrix multiplication
@register_op_test("stablehlo.dot", "stablehlo.dot_general")
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


# Convolution operations
@register_op_test("stablehlo.convolution")
@pytest.mark.parametrize(
    "x, kernel, strides, padding, dilation, groups",
    [
        # Basic 3x3 conv, SAME padding
        (
            np.random.randn(2, 28, 28, 3).astype(np.float32),
            np.random.randn(3, 3, 3, 8).astype(np.float32),
            (1, 1),
            "SAME",
            (1, 1),
            1,
        ),
        # Strided conv
        (
            np.random.randn(2, 32, 32, 3).astype(np.float32),
            np.random.randn(3, 3, 3, 16).astype(np.float32),
            (2, 2),
            "SAME",
            (1, 1),
            1,
        ),
        # VALID padding
        (
            np.random.randn(2, 32, 32, 3).astype(np.float32),
            np.random.randn(5, 5, 3, 8).astype(np.float32),
            (1, 1),
            "VALID",
            (1, 1),
            1,
        ),
        # Dilated conv
        (
            np.random.randn(2, 32, 32, 3).astype(np.float32),
            np.random.randn(3, 3, 3, 8).astype(np.float32),
            (1, 1),
            "SAME",
            (2, 2),
            1,
        ),
        # 1x1 pointwise conv
        (
            np.random.randn(2, 16, 16, 64).astype(np.float32),
            np.random.randn(1, 1, 64, 128).astype(np.float32),
            (1, 1),
            "VALID",
            (1, 1),
            1,
        ),
        # Depthwise conv (groups = in_channels)
        (
            np.random.randn(2, 28, 28, 16).astype(np.float32),
            np.random.randn(3, 3, 1, 16).astype(np.float32),
            (1, 1),
            "SAME",
            (1, 1),
            16,
        ),
        # Grouped conv
        (
            np.random.randn(2, 28, 28, 16).astype(np.float32),
            np.random.randn(3, 3, 4, 32).astype(np.float32),
            (1, 1),
            "SAME",
            (1, 1),
            4,
        ),
    ],
)
@assert_cpu_mps_allclose
def test_conv2d(
    request: pytest.FixtureRequest,
    device,
    x,
    kernel,
    strides,
    padding,
    dilation,
    groups,
):
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=strides,
        padding=padding,
        rhs_dilation=dilation,
        feature_group_count=groups,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


# ReLU activation (uses compare + select internally)
@register_op_test("stablehlo.compare", "stablehlo.select")
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


# Reshape operations
@register_op_test("stablehlo.reshape")
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
@register_op_test("stablehlo.broadcast", "stablehlo.broadcast_in_dim")
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
@register_op_test("stablehlo.convert")
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


# Bitcast conversion (reinterpret memory as different type)
@register_op_test("stablehlo.bitcast_convert")
@pytest.mark.parametrize(
    "x, to_dtype",
    [
        (np.random.randn(16, 16).astype(np.float32), np.int32),
        (np.random.randint(0, 2**31, size=(16, 16)).astype(np.int32), np.float32),
    ],
)
@assert_cpu_mps_allclose
def test_bitcast_convert(request: pytest.FixtureRequest, device, x, to_dtype):
    return jax.lax.bitcast_convert_type(x, to_dtype)


# Clip/clamp operation
@register_op_test("stablehlo.clamp")
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
@register_op_test("stablehlo.transpose")
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
@register_op_test("stablehlo.slice")
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


# Dynamic slice operation (indices not known at compile time)
# xfail: MPS implementation ignores start indices (always slices from 0)
@register_op_test("stablehlo.dynamic_slice")
@pytest.mark.parametrize(
    "shape, start_indices, slice_sizes",
    [
        ((10,), (2,), (4,)),
        ((8, 8), (1, 2), (4, 4)),
        ((4, 8, 8), (1, 2, 0), (2, 4, 4)),
    ],
)
@assert_cpu_mps_allclose
def test_dynamic_slice(
    request: pytest.FixtureRequest, device, shape, start_indices, slice_sizes
):
    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal(shape).astype(np.float32)
    return jax.lax.dynamic_slice(x, start_indices, slice_sizes)


# Concatenate operation
@register_op_test("stablehlo.concatenate")
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
@register_op_test("stablehlo.iota")
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


# Reduce operations (stablehlo.reduce and stablehlo.return are used internally)
@register_op_test("stablehlo.reduce", "stablehlo.return")
@pytest.mark.parametrize(
    "op, x, axis",
    [
        (jnp.sum, np.random.randn(16, 16).astype(np.float32), None),
        (jnp.sum, np.random.randn(8, 4, 2).astype(np.float32), 1),
        (jnp.prod, np.random.randn(4, 4).astype(np.float32) * 0.5 + 1, None),
        (jnp.max, np.random.randn(16, 16).astype(np.float32), 0),
        (jnp.min, np.random.randn(16, 16).astype(np.float32), -1),
        (jnp.all, np.random.rand(8, 8) > 0.5, None),
        (jnp.any, np.random.rand(8, 8) > 0.5, 0),
    ],
)
@assert_cpu_mps_allclose
def test_reduce(request: pytest.FixtureRequest, device, op, x, axis):
    return op(x, axis=axis)


# Gather operation (embedding lookup pattern)
@register_op_test("stablehlo.gather")
@pytest.mark.parametrize(
    "operand, indices",
    [
        # Simple embedding lookup
        (
            np.random.randn(100, 16).astype(np.float32),
            np.array([0, 5, 10, 50, 99], dtype=np.int32),
        ),
        # Batched embedding lookup
        (
            np.random.randn(50, 8).astype(np.float32),
            np.array([[0, 1, 2], [10, 20, 30]], dtype=np.int32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_gather(request: pytest.FixtureRequest, device, operand, indices):
    return jnp.take(operand, indices, axis=0)
