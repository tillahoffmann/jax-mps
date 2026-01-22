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
    for key, value in dtypes_shapes.items():
        if isinstance(value, Callable):
            raise NotImplementedError
        else:
            (shape, dtype) = value
            if dtype is float:
                call_args[key] = random.normal(rngs(), shape)
            else:
                raise ValueError(dtype)

    return module, module(**call_args)


class LogisticRegression(nnx.Module):
    """Simple logistic regression model using Flax NNX."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        logits = self.linear(x)
        return jax.nn.sigmoid(logits)


@pytest.mark.parametrize(
    "x, kernel, bias",
    [
        (
            np.random.randn(32, 16).astype(np.float32),
            np.random.randn(16, 1).astype(np.float32),
            np.random.randn(1).astype(np.float32),
        ),
        (
            np.random.randn(4, 16).astype(np.float32),
            np.random.randn(16, 1).astype(np.float32),
            np.random.randn(1).astype(np.float32),
        ),
    ],
)
@assert_cpu_mps_allclose
def test_linear_sigmoid(request: pytest.FixtureRequest, device, x, kernel, bias):
    """Test that linear + sigmoid produces matching results on CPU and MPS."""
    logits = jnp.matmul(x, kernel) + bias
    return jax.nn.sigmoid(logits)


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
