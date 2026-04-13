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
    make_fused_op_configs,
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
    *make_fused_op_configs(),
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


def test_func_call_no_spurious_errors() -> None:
    """Test that func.call with while loops doesn't produce spurious MPS ERROR messages.

    Regression test for https://github.com/tillahoffmann/jax-mps/issues/91.
    Nested jax.jit calls generate func.call ops in StableHLO. When the callee
    contains stablehlo.while (from lax.scan), the mlx::core::compile() attempt
    fails and falls back to direct execution. This should NOT produce [MPS ERROR]
    messages on stderr.
    """
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import jax
import jax.numpy as jnp
from jax import lax

def inner_fn(x):
    def scan_body(carry, elem):
        return carry + elem, carry + elem
    _, ys = lax.scan(scan_body, jnp.float32(0.0), x)
    return ys

fn = jax.jit(lambda x: jax.jit(inner_fn)(x).sum())
result = fn(jnp.ones((8,)))
result.block_until_ready()
""",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "mps"},
    )
    assert result.returncode == 0, (
        f"Subprocess failed with return code {result.returncode}:\n{result.stderr}"
    )
    assert "[MPS ERROR]" not in result.stderr, (
        f"Spurious MPS ERROR messages in stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "composite_name,decomposition_body,expected_fn",
    [
        pytest.param(
            "chlo.asin",
            # arcsin(x) = atan2(x, sqrt(1 - x*x))
            """
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
    %1 = stablehlo.subtract %cst, %0 : tensor<4xf32>
    %2 = stablehlo.sqrt %1 : tensor<4xf32>
    %3 = stablehlo.atan2 %arg0, %2 : tensor<4xf32>
    return %3 : tensor<4xf32>
""",
            numpy.arcsin,
            id="arcsin",
        ),
        pytest.param(
            "chlo.sinh",
            # sinh(x) = (exp(x) - exp(-x)) / 2
            """
    %0 = stablehlo.exponential %arg0 : tensor<4xf32>
    %1 = stablehlo.negate %arg0 : tensor<4xf32>
    %2 = stablehlo.exponential %1 : tensor<4xf32>
    %3 = stablehlo.subtract %0, %2 : tensor<4xf32>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
    %4 = stablehlo.divide %3, %cst : tensor<4xf32>
    return %4 : tensor<4xf32>
""",
            numpy.sinh,
            id="sinh",
        ),
    ],
)
def test_composite_op(composite_name, decomposition_body, expected_fn) -> None:
    """Test that stablehlo.composite ops execute correctly.

    Regression test for https://github.com/tillahoffmann/jax-mps/issues/95.
    JAX 0.9.1+ wraps CHLO ops (arcsin, sinh, erf, etc.) as stablehlo.composite
    ops instead of dispatching them as custom_call or chlo.* ops directly.
    """
    OperationTestConfig.EXERCISED_STABLEHLO_OPS.add("stablehlo.composite")
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")

    from jax._src import xla_bridge
    from jaxlib import xla_client

    impl_name = f"{composite_name}.impl"
    stablehlo_text = f"""
module @test {{
  func.func private @{impl_name}(%arg0: tensor<4xf32>) -> tensor<4xf32> {{
{decomposition_body}
  }}
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {{
    %0 = stablehlo.composite "{composite_name}" %arg0 {{decomposition = @{impl_name}}} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }}
}}
"""

    client = xla_bridge.get_backend("mps")
    devices = client.local_devices()
    device_list = xla_client.DeviceList(tuple(devices[:1]))
    exe = client.compile_and_load(stablehlo_text.encode(), device_list)

    x = numpy.array([0.0, 0.5, -0.5, 0.9], dtype=numpy.float32)
    buf = client.buffer_from_pyval(x, devices[0])
    result = numpy.asarray(exe.execute([buf])[0])
    expected = expected_fn(x)
    numpy.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


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


@pytest.mark.parametrize(
    "jax_fn,target",
    [
        (jnp.sinh, "mhlo.sinh"),
        (jnp.cosh, "mhlo.cosh"),
        (jnp.arcsin, "mhlo.asin"),
        (jnp.arccos, "mhlo.acos"),
        (jnp.arctan, "mhlo.atan"),
        (jnp.arcsinh, "mhlo.asinh"),
        (jnp.arccosh, "mhlo.acosh"),
        (jnp.arctanh, "mhlo.atanh"),
        (jax.lax.erf, "mhlo.erf"),
        (jax.scipy.special.erfinv, "mhlo.erf_inv"),
    ],
)
def test_unary_ops_lower_to_mhlo_custom_call(jax_fn, target) -> None:
    """Regression: ensure our MPS-platform lowerings in jax_plugins.mps.ops
    keep these ops as stablehlo.custom_call @mhlo.<name> instead of letting
    JAX decompose them through CHLO. A future JAX pipeline change that
    silently reintroduces the decomposition would make this fail."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")

    device = jax.devices("mps")[0]
    with jax.default_device(device):
        x = jnp.ones((3,), dtype=jnp.float32)
        ir_text = str(jax.jit(jax_fn).lower(x).compiler_ir("stablehlo"))
    assert f"@{target}" in ir_text, (
        f"Expected `@{target}` in lowered IR; got:\n{ir_text}"
    )


def test_rng_bit_generator() -> None:
    """Test that stablehlo.rng_bit_generator produces valid output and correct state."""
    OperationTestConfig.EXERCISED_STABLEHLO_OPS.add("stablehlo.rng_bit_generator")
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")

    import jax.numpy as jnp
    from jax import lax

    device = jax.devices("mps")[0]
    with jax.default_device(device):
        state = jnp.array([0, 0, 0, 42], dtype=jnp.uint32)
        new_state, output = lax.rng_bit_generator(state, (8,))

        # Output should have the correct shape and dtype
        assert output.shape == (8,)
        assert output.dtype == jnp.uint32

        # State should be updated (counter incremented)
        assert new_state.shape == state.shape
        assert not jnp.array_equal(new_state, state)

        # Determinism: same input state and shape => same output and next-state.
        new_state2, output2 = lax.rng_bit_generator(state, (8,))
        assert jnp.array_equal(output2, output)
        assert jnp.array_equal(new_state2, new_state)
