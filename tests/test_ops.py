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


# Binary operations
@pytest.mark.parametrize(
    "op",
    [
        register_op_test(jnp.add, "stablehlo.add"),
        register_op_test(jnp.subtract, "stablehlo.subtract"),
        register_op_test(jnp.multiply, "stablehlo.multiply"),
        register_op_test(jnp.divide, "stablehlo.divide"),
    ],
)
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


# Min/max operations
@pytest.mark.parametrize(
    "op",
    [
        register_op_test(jnp.maximum, "stablehlo.maximum"),
        register_op_test(jnp.minimum, "stablehlo.minimum"),
    ],
)
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


# Unary operations
@pytest.mark.parametrize(
    "op, x",
    [
        (
            register_op_test(jnp.tanh, "stablehlo.tanh"),
            np.random.randn(32, 32).astype(np.float32),
        ),
        (
            register_op_test(jnp.exp, "stablehlo.exponential"),
            np.random.randn(32, 32).astype(np.float32) * 0.5,
        ),  # Avoid overflow
        (
            register_op_test(jnp.log, "stablehlo.log"),
            np.abs(np.random.randn(32, 32).astype(np.float32)) + 0.1,
        ),  # Positive values
        (
            register_op_test(jnp.negative, "stablehlo.negate"),
            np.random.randn(32, 32).astype(np.float32),
        ),
        (
            register_op_test(jnp.abs, "stablehlo.abs"),
            np.random.randn(32, 32).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_unary_op(request: pytest.FixtureRequest, device, op, x):
    return op(x)


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


# Additional unary operations
@pytest.mark.parametrize(
    "op, x",
    [
        (
            register_op_test(jnp.sqrt, "stablehlo.sqrt"),
            np.abs(np.random.randn(32, 32).astype(np.float32)) + 0.1,
        ),
        (
            register_op_test(jnp.log1p, "stablehlo.log_plus_one"),
            np.abs(np.random.randn(32, 32).astype(np.float32)),
        ),
        (
            register_op_test(jax.scipy.special.erf, "stablehlo.erf"),
            np.random.randn(32, 32).astype(np.float32) * 0.5,
        ),
        (
            register_op_test(jax.scipy.special.erfinv, "chlo.erf_inv"),
            np.random.uniform(-0.9, 0.9, (32, 32)).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_special_unary_op(request: pytest.FixtureRequest, device, op, x):
    return op(x)


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


# Bitwise operations
@pytest.mark.parametrize(
    "op",
    [
        register_op_test(jnp.bitwise_and, "stablehlo.and"),
        register_op_test(jnp.bitwise_or, "stablehlo.or"),
        register_op_test(jnp.bitwise_xor, "stablehlo.xor"),
    ],
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
    [
        register_op_test(jnp.left_shift, "stablehlo.shift_left"),
        register_op_test(jnp.right_shift, "stablehlo.shift_right_logical"),
    ],
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


# nextafter operation
@register_op_test("chlo.next_after")
@pytest.mark.parametrize(
    "x, y",
    [
        (
            np.array([1.0, -1.0, 0.0, 2.0], dtype=np.float32),
            np.array([2.0, -2.0, 1.0, 1.0], dtype=np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_nextafter(request: pytest.FixtureRequest, device, x, y):
    return jnp.nextafter(x, y)
