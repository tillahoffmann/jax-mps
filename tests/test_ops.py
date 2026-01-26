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

# Common test data (seeded for reproducibility)
_rng = np.random.default_rng(42)
_float_2d = _rng.standard_normal((32, 32)).astype(np.float32)
_float_2d_b = _rng.standard_normal((32, 32)).astype(np.float32) + 0.1
_float_positive = np.abs(_rng.standard_normal((32, 32)).astype(np.float32)) + 0.1
_uint_2d = _rng.integers(0, 256, size=(32, 32)).astype(np.uint32)
_uint_2d_b = _rng.integers(0, 256, size=(32, 32)).astype(np.uint32)
_uint_shift = _rng.integers(0, 8, size=(32, 32)).astype(np.uint32)

# Additional seeded test data for various shapes
_rng2 = np.random.default_rng(43)
_float_1d = _rng2.standard_normal((10,)).astype(np.float32)
_float_2d_small = _rng2.standard_normal((4, 8)).astype(np.float32)
_float_3d = _rng2.standard_normal((2, 3, 4)).astype(np.float32)
_float_4d = _rng2.standard_normal((2, 3, 4, 5)).astype(np.float32)
_float_8x8 = _rng2.standard_normal((8, 8)).astype(np.float32)
_float_4x8x8 = _rng2.standard_normal((4, 8, 8)).astype(np.float32)

# Matmul test data
_rng_matmul = np.random.default_rng(44)
_matmul_a1 = _rng_matmul.standard_normal((32, 64)).astype(np.float32)
_matmul_b1 = _rng_matmul.standard_normal((64, 32)).astype(np.float32)
_matmul_a2 = _rng_matmul.standard_normal((16, 16)).astype(np.float32)
_matmul_b2 = _rng_matmul.standard_normal((16, 16)).astype(np.float32)


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
        (register_op_test(jax.lax.rsqrt, "stablehlo.rsqrt"), _float_positive),
        (register_op_test(jnp.log1p, "stablehlo.log_plus_one"), _float_positive),
        (register_op_test(jax.scipy.special.erf, "stablehlo.erf"), _float_2d * 0.5),
        (
            register_op_test(jax.scipy.special.erfinv, "chlo.erf_inv"),
            _rng.uniform(-0.9, 0.9, (32, 32)).astype(np.float32),
        ),
        (register_op_test(jnp.floor, "stablehlo.floor"), _float_2d),
        (register_op_test(jnp.sign, "stablehlo.sign"), _float_2d),
        (
            register_op_test(jnp.isfinite, "stablehlo.is_finite"),
            np.array([1.0, np.inf, -np.inf, np.nan, 0.0], dtype=np.float32),
        ),
        # ReLU uses compare + select internally
        (
            register_op_test(jax.nn.relu, "stablehlo.compare", "stablehlo.select"),
            _float_2d,
        ),
    ],
)
@assert_cpu_mps_allclose
def test_unary_op(request: pytest.FixtureRequest, device, op, x):
    result = op(x)
    grad = jax.grad(lambda x: op(x).mean())(x)
    return result, grad


# Binary operations (arithmetic, min/max, remainder, bitwise, shifts, matmul)
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
            register_op_test(jnp.power, "stablehlo.power"),
            _float_positive,
            _float_2d_b * 0.5 + 1,
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
        # Matrix multiplication
        (
            register_op_test(jnp.matmul, "stablehlo.dot", "stablehlo.dot_general"),
            _matmul_a1,
            _matmul_b1,
        ),
        (jnp.matmul, _matmul_a2, _matmul_b2),
    ],
)
@assert_cpu_mps_allclose
def test_binary_op(request: pytest.FixtureRequest, device, op, a, b):
    result = op(a, b)

    # Calculate gradients for float types (skip non-differentiable ops)
    if op is jnp.nextafter:
        grad = None
    elif a.dtype == jnp.float32 and b.dtype == jnp.float32:
        grad = jax.grad(lambda x: op(*x).mean())((a, b))
    elif a.dtype == jnp.float32:
        grad = jax.grad(lambda x: op(x, b).mean())(a)
    elif b.dtype == jnp.float32:
        grad = jax.grad(lambda x: op(a, x).mean())(b)
    else:
        grad = None

    return result, grad


# Convolution operations
_rng_conv = np.random.default_rng(45)


@register_op_test("stablehlo.convolution")
@pytest.mark.parametrize(
    "x, kernel, strides, padding, dilation, groups",
    [
        # Basic 3x3 conv, SAME padding
        (
            _rng_conv.standard_normal((2, 28, 28, 3)).astype(np.float32),
            _rng_conv.standard_normal((3, 3, 3, 8)).astype(np.float32),
            (1, 1),
            "SAME",
            (1, 1),
            1,
        ),
        # Strided conv
        (
            _rng_conv.standard_normal((2, 32, 32, 3)).astype(np.float32),
            _rng_conv.standard_normal((3, 3, 3, 16)).astype(np.float32),
            (2, 2),
            "SAME",
            (1, 1),
            1,
        ),
        # VALID padding
        (
            _rng_conv.standard_normal((2, 32, 32, 3)).astype(np.float32),
            _rng_conv.standard_normal((5, 5, 3, 8)).astype(np.float32),
            (1, 1),
            "VALID",
            (1, 1),
            1,
        ),
        # Dilated conv
        (
            _rng_conv.standard_normal((2, 32, 32, 3)).astype(np.float32),
            _rng_conv.standard_normal((3, 3, 3, 8)).astype(np.float32),
            (1, 1),
            "SAME",
            (2, 2),
            1,
        ),
        # 1x1 pointwise conv
        (
            _rng_conv.standard_normal((2, 16, 16, 64)).astype(np.float32),
            _rng_conv.standard_normal((1, 1, 64, 128)).astype(np.float32),
            (1, 1),
            "VALID",
            (1, 1),
            1,
        ),
        # Depthwise conv (groups = in_channels)
        (
            _rng_conv.standard_normal((2, 28, 28, 16)).astype(np.float32),
            _rng_conv.standard_normal((3, 3, 1, 16)).astype(np.float32),
            (1, 1),
            "SAME",
            (1, 1),
            16,
        ),
        # Grouped conv
        (
            _rng_conv.standard_normal((2, 28, 28, 16)).astype(np.float32),
            _rng_conv.standard_normal((3, 3, 4, 32)).astype(np.float32),
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
    def conv_fn(x, kernel):
        return jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=strides,
            padding=padding,
            rhs_dilation=dilation,
            feature_group_count=groups,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    result = conv_fn(x, kernel)
    # Gradient only works for basic convs (stride=1, groups=1)
    # Strided conv gradient produces incorrect values
    # Grouped/depthwise conv gradient crashes with batch_group_count not supported
    if strides == (1, 1) and groups == 1:
        grad = jax.grad(lambda args: conv_fn(*args).mean())((x, kernel))
    else:
        grad = None
    return result, grad


# Shape operations (reshape, broadcast, transpose, reverse)
@pytest.mark.parametrize(
    "op, x, arg",
    [
        # Reshape
        (register_op_test(jnp.reshape, "stablehlo.reshape"), _float_2d, (64, 16)),
        # Broadcast
        (
            register_op_test(
                jnp.broadcast_to, "stablehlo.broadcast", "stablehlo.broadcast_in_dim"
            ),
            _rng.standard_normal((1, 32)).astype(np.float32),
            (4, 32),
        ),
        # Transpose - various ranks
        (register_op_test(jnp.transpose, "stablehlo.transpose"), _float_2d, (1, 0)),
        (jnp.transpose, _float_2d_small, (1, 0)),
        (jnp.transpose, _float_3d, (2, 0, 1)),
        (jnp.transpose, _float_4d, (3, 2, 1, 0)),
        # Reverse - various ranks and axes
        (register_op_test(jax.lax.rev, "stablehlo.reverse"), _float_1d, (0,)),
        (jax.lax.rev, _float_2d_small, (0,)),
        (jax.lax.rev, _float_2d_small, (1,)),
        (jax.lax.rev, _float_2d_small, (0, 1)),
        (jax.lax.rev, _float_3d, (1, 2)),
    ],
)
@assert_cpu_mps_allclose
def test_shape_op(request: pytest.FixtureRequest, device, op, x, arg):
    result = op(x, arg)
    grad = jax.grad(lambda x: op(x, arg).mean())(x)
    return result, grad


# Type conversions (convert and bitcast)
_astype = register_op_test(lambda x, d: x.astype(d), "stablehlo.convert")


@pytest.mark.parametrize(
    "op, x, to_dtype",
    [
        # Regular conversions
        (_astype, _float_2d, np.float16),
        (_astype, _float_2d.astype(np.float16), np.float32),
        (_astype, _rng.integers(-100, 100, (16, 16)).astype(np.int32), np.float32),
        # Bitcast conversions
        (
            register_op_test(jax.lax.bitcast_convert_type, "stablehlo.bitcast_convert"),
            _float_2d,
            np.int32,
        ),
    ],
)
@assert_cpu_mps_allclose
def test_type_convert(request: pytest.FixtureRequest, device, op, x, to_dtype):
    return op(x, to_dtype)


# Clip/clamp operation
@register_op_test("stablehlo.clamp")
@pytest.mark.parametrize(
    "x, a_min, a_max",
    [
        (_float_2d, -0.5, 0.5),
        (_float_2d_small, 0.0, 1.0),
    ],
)
@assert_cpu_mps_allclose
def test_clip(request: pytest.FixtureRequest, device, x, a_min, a_max):
    result = jnp.clip(x, a_min, a_max)
    grad = jax.grad(lambda x: jnp.clip(x, a_min, a_max).mean())(x)
    return result, grad


# Slicing operations (static and dynamic)
@register_op_test("stablehlo.slice")
@pytest.mark.parametrize(
    "x, slices",
    [
        (_float_1d, (slice(2, 8),)),
        (_float_8x8, (slice(1, 5), slice(2, 6))),
        (_float_4x8x8, (slice(1, 3), slice(2, 6), slice(0, 4))),
    ],
)
@assert_cpu_mps_allclose
def test_slice(request: pytest.FixtureRequest, device, x, slices):
    def do_slice(x):
        return x[slices]

    result = do_slice(x)
    grad = jax.grad(lambda x: do_slice(x).mean())(x)
    return result, grad


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
    result = jax.lax.dynamic_slice(x, start_indices, slice_sizes)
    grad = jax.grad(
        lambda x: jax.lax.dynamic_slice(x, start_indices, slice_sizes).mean()
    )(x)
    return result, grad


# Concatenate operation
_rng_concat = np.random.default_rng(46)


@register_op_test("stablehlo.concatenate")
@pytest.mark.parametrize(
    "arrays, axis",
    [
        ([_rng_concat.standard_normal((4, 8)).astype(np.float32) for _ in range(3)], 0),
        ([_rng_concat.standard_normal((4, 8)).astype(np.float32) for _ in range(2)], 1),
        (
            [
                _rng_concat.standard_normal((2, 3, 4)).astype(np.float32)
                for _ in range(2)
            ],
            2,
        ),
    ],
)
@assert_cpu_mps_allclose
def test_concatenate(request: pytest.FixtureRequest, device, arrays, axis):
    result = jnp.concatenate(arrays, axis=axis)
    grad = jax.grad(lambda arrs: jnp.concatenate(arrs, axis=axis).mean())(arrays)
    return result, grad


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


# Reduce operations (stablehlo.reduce and stablehlo.return are used internally)
_rng_reduce = np.random.default_rng(47)


@register_op_test("stablehlo.reduce", "stablehlo.return")
@pytest.mark.parametrize(
    "op, x, axis",
    [
        (jnp.sum, _rng_reduce.standard_normal((16, 16)).astype(np.float32), None),
        (jnp.sum, _rng_reduce.standard_normal((8, 4, 2)).astype(np.float32), 1),
        (
            jnp.prod,
            _rng_reduce.standard_normal((4, 4)).astype(np.float32) * 0.5 + 1,
            None,
        ),
        (jnp.max, _rng_reduce.standard_normal((16, 16)).astype(np.float32), 0),
        (jnp.min, _rng_reduce.standard_normal((16, 16)).astype(np.float32), -1),
        (jnp.all, _rng_reduce.random((8, 8)) > 0.5, None),
        (jnp.any, _rng_reduce.random((8, 8)) > 0.5, 0),
    ],
)
@assert_cpu_mps_allclose
def test_reduce(request: pytest.FixtureRequest, device, op, x, axis):
    result = op(x, axis=axis)
    # Only compute gradients for differentiable reduce ops with float inputs
    if op in (jnp.sum, jnp.prod) and x.dtype == np.float32:
        grad = jax.grad(lambda x: op(x, axis=axis).sum())(x)
    else:
        grad = None
    return result, grad


# Gather operation (embedding lookup pattern)
_rng_gather = np.random.default_rng(48)


@register_op_test("stablehlo.gather")
@pytest.mark.parametrize(
    "operand, indices",
    [
        # Simple embedding lookup (1D indices - gradient works)
        (
            _rng_gather.standard_normal((100, 16)).astype(np.float32),
            np.array([0, 5, 10, 50, 99], dtype=np.int32),
        ),
        # Batched embedding lookup (2D indices - gradient crashes MPS scatter)
        (
            _rng_gather.standard_normal((50, 8)).astype(np.float32),
            np.array([[0, 1, 2], [10, 20, 30]], dtype=np.int32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_gather(request: pytest.FixtureRequest, device, operand, indices):
    result = jnp.take(operand, indices, axis=0)
    # 2D indices gradient crashes MPS with scatter rank mismatch error
    if indices.ndim == 1:
        grad = jax.grad(lambda x: jnp.take(x, indices, axis=0).mean())(operand)
    else:
        grad = None
    return result, grad


# Pad operation
_rng_pad = np.random.default_rng(49)


@register_op_test("stablehlo.pad")
@pytest.mark.parametrize(
    "x, pad_width, constant_value",
    [
        (_rng_pad.standard_normal((4, 4)).astype(np.float32), ((1, 1), (2, 2)), 0.0),
        (_rng_pad.standard_normal((3, 5)).astype(np.float32), ((0, 2), (1, 0)), 1.0),
        (_rng_pad.standard_normal((8,)).astype(np.float32), ((3, 3),), -1.0),
    ],
)
@assert_cpu_mps_allclose
def test_pad(request: pytest.FixtureRequest, device, x, pad_width, constant_value):
    # Gradient crashes MPS with strided_slice_update shape mismatch
    return jnp.pad(x, pad_width, constant_values=constant_value)


# Dynamic update slice operation
@register_op_test("stablehlo.dynamic_update_slice")
@pytest.mark.parametrize(
    "operand, update, start_indices",
    [
        (
            np.zeros((8,), dtype=np.float32),
            np.ones((3,), dtype=np.float32),
            (2,),
        ),
        (
            np.zeros((6, 6), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            (1, 2),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_dynamic_update_slice(
    request: pytest.FixtureRequest, device, operand, update, start_indices
):
    result = jax.lax.dynamic_update_slice(operand, update, start_indices)
    grad = jax.grad(
        lambda x: jax.lax.dynamic_update_slice(operand, x, start_indices).mean()
    )(update)
    return result, grad


# Scatter operation
@register_op_test("stablehlo.scatter")
@pytest.mark.parametrize(
    "operand, indices, updates",
    [
        # Simple scatter add
        (
            np.zeros((10, 4), dtype=np.float32),
            np.array([0, 2, 5], dtype=np.int32),
            np.ones((3, 4), dtype=np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_scatter(request: pytest.FixtureRequest, device, operand, indices, updates):
    # Gradient fails with "Memory kinds and dtypes have different sizes" error
    return operand.at[indices].add(updates)


# Non-contiguous array transfer (regression test for CIFAR loader bug)
# Create non-contiguous test data - transpose creates view with non-standard strides
_noncontig_rng = np.random.default_rng(42)
_noncontig_array = (
    _noncontig_rng.standard_normal((256, 3, 32, 32))
    .astype(np.float32)
    .transpose(0, 2, 3, 1)
)
assert not _noncontig_array.flags["C_CONTIGUOUS"], "Test data must be non-contiguous"


@pytest.mark.xfail(reason="MPS corrupts non-contiguous arrays during transfer")
@pytest.mark.parametrize("x", [_noncontig_array])
@assert_cpu_mps_allclose
def test_noncontiguous_array_transfer(request: pytest.FixtureRequest, device, x):
    """Test that non-contiguous arrays are transferred correctly to MPS.

    This is a regression test for a bug where transpose() created a non-contiguous
    array that was corrupted when transferred to MPS, causing training divergence.
    """
    # The decorator already transfers x to the device via jax.device_put.
    # Just return it - the comparison will catch corruption.
    return x
