import os
import re
from pathlib import Path

import jax
import numpy
import pytest
from jax import dtypes, random

from .configs import (
    OperationTestConfig,
    make_binary_op_configs,
    make_conv_op_configs,
    make_conversion_op_configs,
    make_flax_op_configs,
    make_linalg_op_configs,
    make_misc_op_configs,
    make_random_op_configs,
    make_reduction_op_configs,
    make_shape_op_configs,
    make_slice_op_configs,
    make_unary_op_configs,
)

OPERATION_TEST_CONFIGS = [
    *make_binary_op_configs(),
    *make_conv_op_configs(),
    *make_conversion_op_configs(),
    *make_flax_op_configs(),
    *make_linalg_op_configs(),
    *make_misc_op_configs(),
    *make_random_op_configs(),
    *make_reduction_op_configs(),
    *make_shape_op_configs(),
    *make_slice_op_configs(),
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
        numpy.testing.assert_allclose(actual, desired, atol=1e-5, rtol=1e-5)
    except AssertionError as ex:
        raise AssertionError(f"Values are not close at path '{path}'.") from ex


def test_op_value(op_config: OperationTestConfig, jit: bool) -> None:
    results = []
    for platform in ["cpu", "mps"]:
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

    jax.tree.map_with_path(assert_allclose_with_path, *results)


def test_op_grad(op_config: OperationTestConfig, jit: bool) -> None:
    argnums = op_config.get_differentiable_argnums()
    if not argnums:
        pytest.skip(f"No differentiable arguments for operation '{op_config.func}'.")

    for argnum in argnums:
        results = []
        for platform in ["cpu", "mps"]:
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

        jax.tree.map_with_path(assert_allclose_with_path, *results)


def test_unsupported_op_error_message(jit: bool) -> None:
    """Check that unsupported-op errors link to the issue template and CONTRIBUTING.md."""
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


@pytest.fixture(autouse=True, scope="module")
def assert_all_ops_tested():
    yield

    if "CI" not in os.environ:
        return

    ops_dir = Path(__file__).parent.parent / "src/pjrt_plugin/ops"
    assert ops_dir.is_dir()

    # Patterns matching op registration calls
    patterns = [
        re.compile(r'REGISTER_MPS_OP\("([^"]+)"'),
        re.compile(r'REGISTER_MLIR_BINARY_OP\("([^"]+)"'),
        re.compile(r'REGISTER_MLIR_UNARY_OP\("([^"]+)"'),
        re.compile(r'REGISTER_LOGICAL_BITWISE_OP\("([^"]+)"'),
        re.compile(r'OpRegistry::Register\("([^"]+)"'),
    ]

    op_names = set()
    for mm_file in ops_dir.glob("*.mm"):
        with mm_file.open() as fp:
            content = fp.read()
            for pattern in patterns:
                op_names.update(pattern.findall(content))

    assert op_names, "Failed to discover any ops."
    unsupported = OperationTestConfig.EXERCISED_STABLEHLO_OPS - op_names
    assert not unsupported, (
        f"Discovered {len(unsupported)} unsupported ops: {', '.join(sorted(unsupported))}"
    )
    missing = op_names - OperationTestConfig.EXERCISED_STABLEHLO_OPS
    assert not missing, (
        f"Discovered {len(missing)} untested ops: {', '.join(sorted(missing))}"
    )
