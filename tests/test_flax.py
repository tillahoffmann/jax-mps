"""Tests for Flax NNX models comparing CPU vs MPS (Metal) results."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose


class LogisticRegression(nnx.Module):
    """Simple logistic regression model using Flax NNX."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        logits = self.linear(x)
        return jax.nn.sigmoid(logits)


def test_logistic_regression_cpu_vs_mps(jax_setup):
    """Test that logistic regression produces matching results on CPU and MPS."""
    cpu = jax_setup["cpu"]
    mps = jax_setup["mps"]

    seed = 42
    in_features = 16
    out_features = 1
    batch_size = 32

    np.random.seed(seed)
    x_np = np.random.randn(batch_size, in_features).astype(np.float32)

    # Create model on CPU (MPS doesn't support RNG operations for initialization)
    with jax.default_device(cpu):
        model = LogisticRegression(in_features, out_features, rngs=nnx.Rngs(seed))
        kernel_np = np.array(model.linear.kernel[...])
        bias_np = np.array(model.linear.bias[...])

    # Run on CPU
    x_cpu = jax.device_put(x_np, cpu)
    kernel_cpu = jax.device_put(kernel_np, cpu)
    bias_cpu = jax.device_put(bias_np, cpu)
    logits_cpu = jnp.matmul(x_cpu, kernel_cpu) + bias_cpu
    output_cpu = jax.nn.sigmoid(logits_cpu)

    # Run on MPS with same weights
    x_mps = jax.device_put(x_np, mps)
    kernel_mps = jax.device_put(kernel_np, mps)
    bias_mps = jax.device_put(bias_np, mps)
    logits_mps = jnp.matmul(x_mps, kernel_mps) + bias_mps
    output_mps = jax.nn.sigmoid(logits_mps)

    assert_allclose(
        np.array(output_mps),
        np.array(output_cpu),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Logistic regression output differs between CPU and MPS",
    )
