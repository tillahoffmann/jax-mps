"""Tests for Flax NNX models comparing CPU vs MPS (Metal) results."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_cpu_mps_allclose
from flax import nnx
from jax import random


@pytest.mark.parametrize(
    "cls, args, dtypes_shapes",
    [
        (
            nnx.Linear,
            {"in_features": 3, "out_features": 4},
            {"inputs": ((10, 3), float)},
        ),
        # Basic 2D conv with SAME padding
        (
            nnx.Conv,
            {"in_features": 3, "out_features": 8, "kernel_size": (3, 3)},
            {"inputs": ((4, 28, 28, 3), float)},
        ),
        # Strided convolution
        (
            nnx.Conv,
            {
                "in_features": 3,
                "out_features": 16,
                "kernel_size": (3, 3),
                "strides": (2, 2),
            },
            {"inputs": ((2, 32, 32, 3), float)},
        ),
        # Valid padding (no padding)
        (
            nnx.Conv,
            {
                "in_features": 3,
                "out_features": 8,
                "kernel_size": (5, 5),
                "padding": "VALID",
            },
            {"inputs": ((2, 32, 32, 3), float)},
        ),
        # Dilated convolution
        (
            nnx.Conv,
            {
                "in_features": 3,
                "out_features": 8,
                "kernel_size": (3, 3),
                "kernel_dilation": (2, 2),
            },
            {"inputs": ((2, 32, 32, 3), float)},
        ),
        # 1x1 convolution (pointwise)
        (
            nnx.Conv,
            {"in_features": 64, "out_features": 128, "kernel_size": (1, 1)},
            {"inputs": ((2, 16, 16, 64), float)},
        ),
        # Depthwise convolution (feature_group_count = in_features)
        pytest.param(
            nnx.Conv,
            {
                "in_features": 16,
                "out_features": 16,
                "kernel_size": (3, 3),
                "feature_group_count": 16,
            },
            {"inputs": ((2, 28, 28, 16), float)},
            marks=pytest.mark.xfail(reason="MPS: batch_group_count != 1 not supported"),
        ),
        # Grouped convolution
        pytest.param(
            nnx.Conv,
            {
                "in_features": 16,
                "out_features": 32,
                "kernel_size": (3, 3),
                "feature_group_count": 4,
            },
            {"inputs": ((2, 28, 28, 16), float)},
            marks=pytest.mark.xfail(reason="MPS: batch_group_count != 1 not supported"),
        ),
        # Strided + dilated + valid padding combined
        (
            nnx.Conv,
            {
                "in_features": 8,
                "out_features": 16,
                "kernel_size": (3, 3),
                "strides": (2, 2),
                "kernel_dilation": (2, 2),
                "padding": "VALID",
            },
            {"inputs": ((2, 32, 32, 8), float)},
        ),
        (
            nnx.Embed,
            {
                "num_embeddings": 100,
                "features": 5,
            },
            {"inputs": ((3, 4), int)},
        ),
        # BatchNorm
        (
            nnx.BatchNorm,
            {"num_features": 16, "momentum": 0.9, "epsilon": 1e-5},
            {"x": ((4, 16), float)},
        ),
        # BatchNorm with spatial dimensions (like in CNN)
        (
            nnx.BatchNorm,
            {"num_features": 8, "momentum": 0.9, "epsilon": 1e-5},
            {"x": ((2, 28, 28, 8), float)},
        ),
    ],
)
@assert_cpu_mps_allclose
def test_flax_modules(
    request,
    device,
    cls: type[nnx.Module],
    args: dict[str, Any],
    dtypes_shapes: dict[str, tuple[tuple[int], Any] | Callable],
):
    rngs = nnx.Rngs(42)
    args = args.copy()
    args.setdefault("rngs", rngs)
    module = cls(**args)

    call_args = {}
    has_float_input = False
    for key, value in dtypes_shapes.items():
        if isinstance(value, Callable):
            raise NotImplementedError
        else:
            (shape, dtype) = value
            if dtype is float:
                call_args[key] = random.normal(rngs(), shape)
                has_float_input = True
            elif dtype is int:
                call_args[key] = random.randint(rngs(), shape, 0, 10)
            else:
                raise ValueError(dtype)

    result = module(**call_args)

    # Compute gradients w.r.t. all parameters for differentiable modules
    if has_float_input:

        def loss_fn(model):
            return model(**call_args).mean()

        grads = nnx.grad(loss_fn)(module)
        return result, grads
    else:
        return result


class LogisticRegression(nnx.Module):
    """Simple logistic regression model using Flax NNX."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        logits = self.linear(x)
        return jax.nn.sigmoid(logits)


_linear_sigmoid_rng = np.random.default_rng(42)


@pytest.mark.parametrize(
    "x, kernel, bias",
    [
        (
            _linear_sigmoid_rng.standard_normal((32, 16)).astype(np.float32),
            _linear_sigmoid_rng.standard_normal((16, 1)).astype(np.float32),
            _linear_sigmoid_rng.standard_normal((1,)).astype(np.float32),
        ),
        (
            _linear_sigmoid_rng.standard_normal((4, 16)).astype(np.float32),
            _linear_sigmoid_rng.standard_normal((16, 1)).astype(np.float32),
            _linear_sigmoid_rng.standard_normal((1,)).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_linear_sigmoid(request: pytest.FixtureRequest, device, x, kernel, bias):
    """Test that linear + sigmoid produces matching results on CPU and MPS."""

    def forward(x, kernel, bias):
        logits = jnp.matmul(x, kernel) + bias
        return jax.nn.sigmoid(logits)

    result = forward(x, kernel, bias)
    # Compute gradients w.r.t. all inputs
    grad_fn = jax.grad(lambda args: forward(*args).mean())
    grads = grad_fn((x, kernel, bias))
    return result, grads


def test_flax_model_init(device):
    """Test that Flax NNX models can be initialized and run on each device."""
    with jax.default_device(device):
        model = LogisticRegression(16, 1, rngs=nnx.Rngs(42))
        x = np.random.randn(4, 16).astype(np.float32)
        result = model(x)

    # Just verify it runs and produces valid output (can't compare CPU/MPS with random weights)
    assert result.shape == (4, 1)
    result_np = np.array(result)
    assert np.all(result_np >= 0.0) and np.all(result_np <= 1.0), (
        "Sigmoid output out of range"
    )
