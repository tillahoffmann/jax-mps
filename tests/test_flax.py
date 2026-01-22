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

    # Create model on CPU for reference
    with jax.default_device(cpu):
        model_cpu = LogisticRegression(in_features, out_features, rngs=nnx.Rngs(seed))
        kernel_cpu_np = np.array(model_cpu.linear.kernel[...])
        bias_cpu_np = np.array(model_cpu.linear.bias[...])

    # Run on CPU
    x_cpu = jax.device_put(x_np, cpu)
    logits_cpu = jnp.matmul(x_cpu, model_cpu.linear.kernel) + model_cpu.linear.bias
    output_cpu = jax.nn.sigmoid(logits_cpu)

    # Run on MPS with same weights (transfer from CPU)
    x_mps = jax.device_put(x_np, mps)
    kernel_mps = jax.device_put(kernel_cpu_np, mps)
    bias_mps = jax.device_put(bias_cpu_np, mps)
    logits_mps = jnp.matmul(x_mps, kernel_mps) + bias_mps
    output_mps = jax.nn.sigmoid(logits_mps)

    assert_allclose(
        np.array(output_mps),
        np.array(output_cpu),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Logistic regression output differs between CPU and MPS",
    )


def test_flax_init_on_mps(jax_setup):
    """Test that Flax models can be initialized directly on MPS."""
    mps = jax_setup["mps"]

    seed = 42
    in_features = 16
    out_features = 1

    # Initialize model directly on MPS
    with jax.default_device(mps):
        model = LogisticRegression(in_features, out_features, rngs=nnx.Rngs(seed))

    # Verify model is on MPS
    assert mps in model.linear.kernel.devices(), "Kernel should be on MPS"
    assert mps in model.linear.bias.devices(), "Bias should be on MPS"

    # Verify shapes
    assert model.linear.kernel.shape == (in_features, out_features)
    assert model.linear.bias.shape == (out_features,)

    # Verify we can do a forward pass
    x = jax.device_put(np.random.randn(4, in_features).astype(np.float32), mps)
    output = model(x)
    assert output.shape == (4, out_features)
    assert mps in output.devices(), "Output should be on MPS"
