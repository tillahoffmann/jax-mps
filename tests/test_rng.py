"""Tests for random number generation on MPS."""

import jax
import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(10,), (32,), (8, 8)])
def test_rng_uniform(device, shape):
    """Test that jax.random.uniform works and produces valid values."""
    key = jax.random.key(42)
    with jax.default_device(device):
        result = jax.random.uniform(key, shape=shape)

    result_np = np.array(result)
    assert result.shape == shape
    assert np.all(result_np >= 0.0), f"Got values < 0: {result_np}"
    assert np.all(result_np <= 1.0), f"Got values > 1: {result_np}"


@pytest.mark.parametrize("shape", [(100,), (32, 32)])
def test_rng_normal(device, shape):
    """Test that jax.random.normal works and produces valid distribution."""
    key = jax.random.key(42)
    with jax.default_device(device):
        result = jax.random.normal(key, shape=shape)

    result_np = np.array(result)
    assert result.shape == shape
    # Mean should be close to 0, std close to 1 for large samples
    assert np.abs(np.mean(result_np)) < 0.5, (
        f"Mean too far from 0: {np.mean(result_np)}"
    )
    assert np.abs(np.std(result_np) - 1.0) < 0.5, (
        f"Std too far from 1: {np.std(result_np)}"
    )


def test_rng_split(device):
    """Test that jax.random.split works."""
    key = jax.random.key(42)
    with jax.default_device(device):
        key1, key2 = jax.random.split(key)

    key1_np = np.array(key1._base_array)
    key2_np = np.array(key2._base_array)
    assert key1_np.shape == (2,), f"Expected key shape (2,), got {key1_np.shape}"
    assert key2_np.shape == (2,), f"Expected key shape (2,), got {key2_np.shape}"
    assert not np.array_equal(key1_np, key2_np), "Keys should be different"
