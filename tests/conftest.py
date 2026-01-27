"""Shared pytest fixtures and utilities for MPS tests."""

import functools
from collections.abc import Callable

import jax
import numpy as np
import pytest


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    return jax.devices(request.param)[0]


def _assert(cond, message):
    """Functional assert that can be used in lambdas."""
    assert cond, message


def assert_cpu_mps_allclose(func: Callable):
    """Decorator that runs a test on both CPU and MPS, then compares results.

    The decorated function must:
    - Take `request` and `device` as first two arguments
    - Return a JAX array result
    - Use the `device` fixture (parameterized for cpu/mps)

    Results from CPU and MPS runs are compared with np.testing.assert_allclose.
    """
    values = {}

    @functools.wraps(func)
    def _wrapper(request: pytest.FixtureRequest, device, *args, **kwargs) -> None:
        # Prepare all the keys and validate current state.
        assert f"[{device.platform}-" in request.node.name
        key = request.node.name.replace(f"[{device.platform}-", "[*-")
        current = values.setdefault(key, {})
        assert device.platform not in current, (
            f"Result for key '{key}' on platform '{device.platform}' already exists."
        )

        # Move numpy arrays to the target device explicitly.
        args = [
            jax.device_put(x, device) if isinstance(x, np.ndarray) else x for x in args
        ]
        kwargs = {
            k: jax.device_put(v, device) if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        # Use default_device for Python scalars and any arrays created during execution.
        with jax.default_device(device):
            result = func(request, device, *args, **kwargs)
        assert result is not None, f"Decorated function {func} must return a value."

        jax.tree.map(
            lambda x: _assert(
                x.device == device,
                f"Result is on device '{x.device}'; expected '{device}'.",
            ),
            result,
        )

        # Store results and verify.
        current[device.platform] = result
        if len(current) == 2:
            # Use rtol=2e-5, atol=5e-6 to accommodate CPU/MPS floating point differences.
            # atol is needed because near-zero values can have large relative differences.
            # Large kernel convolutions (7x7) with strides can accumulate ~3e-6 differences.
            jax.tree.map(
                lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-5, atol=5e-6),
                *current.values(),
            )
        elif len(current) > 2:
            raise ValueError

    return _wrapper
