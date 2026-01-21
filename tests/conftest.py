"""Shared pytest fixtures and utilities for MPS tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_plugins import mps


def _check_mps_working():
    """Check if MPS device is available and operations actually work."""
    try:
        mps.initialize()
        mps_device = jax.devices("mps")[0]

        # Simple sanity check: does add actually compute?
        a = jax.device_put(jnp.array([1.0, 2.0, 3.0]), mps_device)
        b = jax.device_put(jnp.array([4.0, 5.0, 6.0]), mps_device)
        result = a + b
        expected = np.array([5.0, 7.0, 9.0])

        if not np.allclose(np.array(result), expected, rtol=1e-5):
            return False, "MPS operations not computing correctly"

        return True, None
    except Exception as e:
        return False, str(e)


_MPS_WORKING, _MPS_ERROR = _check_mps_working()


@pytest.fixture(scope="module")
def jax_setup():
    """Set up JAX with MPS plugin."""
    if not _MPS_WORKING:
        pytest.skip(f"MPS plugin not working: {_MPS_ERROR}")

    mps.initialize()

    return {
        "jax": jax,
        "jnp": jnp,
        "cpu": jax.devices("cpu")[0],
        "mps": jax.devices("mps")[0],
    }


def assert_on_mps(result, mps_device):
    """Assert that the result tensor is on the MPS device."""
    result_device = result.devices()
    assert mps_device in result_device, (
        f"Result is on {result_device}, expected MPS device {mps_device}. "
        "Operation may have fallen back to CPU."
    )
