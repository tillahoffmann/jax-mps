import os

import jax
import numpy
import pytest
from jax import dtypes, random
from jax import numpy as jnp

from .configs import (
    OperationTestConfig,
    make_binary_op_configs,
    make_control_flow_op_configs,
    make_conv_op_configs,
    make_conversion_op_configs,
    make_flax_op_configs,
    make_linalg_op_configs,
    make_matmul_op_configs,
    make_misc_op_configs,
    make_numpyro_op_configs,
    make_random_op_configs,
    make_reduction_op_configs,
    make_shape_op_configs,
    make_slice_op_configs,
    make_sort_op_configs,
    make_unary_op_configs,
)

# Test mode configuration via environment variable:
# - "compare" (default): Run on both CPU and MPS, compare results
# - "mps": Run only on MPS
# - "cpu": Run only on CPU
TEST_MODE = os.environ.get("JAX_TEST_MODE", "compare").lower()
if TEST_MODE not in ("compare", "mps", "cpu"):
    raise ValueError(
        f"Invalid JAX_TEST_MODE: {TEST_MODE}. Must be 'compare', 'mps', or 'cpu'."
    )


def get_test_platforms() -> list[str]:
    """Return the platforms to test based on JAX_TEST_MODE environment variable."""
    if TEST_MODE == "compare":
        return ["cpu", "mps"]
    else:
        return [TEST_MODE]


OPERATION_TEST_CONFIGS = [
    *make_binary_op_configs(),
    *make_control_flow_op_configs(),
    *make_conv_op_configs(),
    *make_conversion_op_configs(),
    *make_flax_op_configs(),
    *make_linalg_op_configs(),
    *make_matmul_op_configs(),
    *make_misc_op_configs(),
    *make_numpyro_op_configs(),
    *make_random_op_configs(),
    *make_reduction_op_configs(),
    *make_shape_op_configs(),
    *make_slice_op_configs(),
    *make_sort_op_configs(),
    *make_unary_op_configs(),
]


@pytest.fixture(params=OPERATION_TEST_CONFIGS, ids=lambda op_config: op_config.name)
def op_config(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[True, False], ids=["jit", "eager"])
def jit(request: pytest.FixtureRequest):
    return request.param


def fassert(cond: bool, message: str) -> None:
    """Functional assertion."""
    assert cond, message


def assert_allclose_with_path(path, actual, desired):
    # Extract key data if these are random keys rather than regular data.
    is_prng_key = dtypes.issubdtype(actual.dtype, dtypes.prng_key)  # pyright: ignore[reportPrivateImportUsage]
    if is_prng_key:
        actual = random.key_data(actual)
        desired = random.key_data(desired)

    try:
        # Use exact comparison for exact dtypes, tolerance-based for inexact.
        if jnp.issubdtype(actual.dtype, jnp.inexact):
            numpy.testing.assert_allclose(actual, desired, atol=1e-5, rtol=1e-5)
        else:
            numpy.testing.assert_array_equal(actual, desired)
    except AssertionError as ex:
        raise AssertionError(f"Values are not close at path '{path}'.") from ex


def test_op_value(op_config: OperationTestConfig, jit: bool) -> None:
    platforms = get_test_platforms()
    results = []
    for platform in platforms:
        device = jax.devices(platform)[0]
        with jax.default_device(device):
            result = op_config.evaluate_value(jit)
            jax.tree.map_with_path(
                lambda path, value: fassert(
                    value.device == device,
                    f"Value at '{path}' is on device {value.device}; expected {device}.",
                ),
                result,
            )
            results.append(result)

    if len(results) == 2:
        jax.tree.map_with_path(assert_allclose_with_path, *results)


def test_op_grad(
    op_config: OperationTestConfig, jit: bool, request: pytest.FixtureRequest
) -> None:
    argnums = op_config.get_differentiable_argnums()
    if not argnums:
        pytest.skip(f"No differentiable arguments for operation '{op_config.func}'.")

    if op_config.grad_xfail:
        from .configs.util import MPS_DEVICE

        # Only apply xfail when actually testing MPS, not just when MPS is available
        if MPS_DEVICE is not None and "mps" in get_test_platforms():
            request.applymarker(
                pytest.mark.xfail(  # type: ignore[call-overload]
                    reason=op_config.grad_xfail,
                    match=op_config.grad_xfail,
                    strict=True,
                )
            )

    platforms = get_test_platforms()
    for argnum in argnums:
        results = []
        for platform in platforms:
            device = jax.devices(platform)[0]
            with jax.default_device(device):
                result = op_config.evaluate_grad(argnum, jit)
                jax.tree.map_with_path(
                    lambda path, value: fassert(
                        value.device == device,
                        f"Value at '{path}' is on device {value.device}; expected {device}.",
                    ),
                    result,
                )
                results.append(result)

        if len(results) == 2:
            jax.tree.map_with_path(assert_allclose_with_path, *results)


def test_unsupported_op_error_message(jit: bool) -> None:
    """Check that unsupported-op errors link to the issue template and CONTRIBUTING.md."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    device = jax.devices("mps")[0]
    with jax.default_device(device):
        try:
            # This is an obscure op. It's unlikely to be implemented, but this test may
            # break if `clz` gets implemented.
            func = jax.lax.clz
            if jit:
                func = jax.jit(func)
            func(numpy.int32(7))
        except Exception as exc:
            message = str(exc)
            assert "issues/new?template=missing-op.yml" in message
            assert "CONTRIBUTING.md" in message
        else:
            pytest.skip("clz is now supported; test needs a new unregistered op")
