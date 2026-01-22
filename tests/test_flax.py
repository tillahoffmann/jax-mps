"""Tests for Flax NNX models comparing CPU vs MPS (Metal) results."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_cpu_mps_allclose


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
