import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import numpy
import pytest
from jax import dtypes, lax, random
from jax import numpy as jnp
from jax.scipy import special

CPU_DEVICE = jax.devices("cpu")[0]
MPS_DEVICE = jax.devices("mps")[0]


_STABLEHLO_OP_RE = re.compile(r"(?<![\#\!])(?:stablehlo|chlo)\.[\w\.]+")


def get_device_placement(value):
    device = None
    for leaf in jax.tree.flatten(value)[0]:
        if not isinstance(leaf, jax.Array):
            continue
        assert device is None or device == leaf.device, "Mixed device placement."
        device = leaf.device
    assert device is not None, "Failed to infer device placement."
    return device


class OperationTestConfig:
    """Configuration for testing operations.

    Args:
        op: Operation to test.
        *args: Factory functions for positional arguments. Non-callables will
            automatically be wrapped in lambdas.
        **kwargs: Factory functions for keyword arguments. Non-callables will
            automatically be wrapped in lambdas.
        differentiable_argnums: Position of arguments that can be differentiated with
            respect to. Defaults to positional arguments with inexact types.
        static_argnums: Position of arguments that should be treated as static in
            jit-compile.
    """

    EXERCISED_STABLEHLO_OPS: set[str] = {
        # HACK: Register these ops as exercised because JAX doesn't seem to generate them.
        "stablehlo.broadcast",
        "stablehlo.dot",
        "stablehlo.erf",
    }

    def __init__(
        self,
        op: Callable,
        *args: Any,
        differentiable_argnums: Sequence[int] | None = None,
        static_argnums: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self.op = op
        self.differentiable_argnums = differentiable_argnums
        self.static_argnums = static_argnums
        self.args = [arg if callable(arg) else lambda arg=arg: arg for arg in args]
        self.kwargs = {
            key: arg if callable(arg) else lambda arg=arg: arg
            for key, arg in kwargs.items()
        }

    def get_args(self):
        """Get positional arguments."""
        args = []
        for arg_func in self.args:
            arg = arg_func()
            if isinstance(arg, numpy.ndarray):
                arg = jnp.asarray(arg)
            args.append(arg)
        return args

    def get_kwargs(self):
        """Get keyword arguments."""
        return {key: arg() for key, arg in self.kwargs.items()}

    def get_differentiable_argnums(self) -> tuple[int, ...]:
        """Get a tuple of integers indicating which arguments can be differentiated with
        respect to."""
        if self.differentiable_argnums is not None:
            return tuple(self.differentiable_argnums)

        differentiable_argnums: list[int] = []
        for argnum, arg in enumerate(self.get_args()):
            if isinstance(arg, float):
                differentiable_argnums.append(argnum)
            elif isinstance(arg, jnp.ndarray):
                if arg.dtype == jnp.float32:
                    differentiable_argnums.append(argnum)
        return tuple(differentiable_argnums)

    def evaluate_value(self, jit: bool):
        """Evaluate the output of the operation."""
        op = self.op
        args = self.get_args()
        kwargs = self.get_kwargs()
        lowered = None
        if jit:
            op = jax.jit(op, static_argnums=self.static_argnums)
            lowered = op.lower(*args, **kwargs)
        result = op(*args, **kwargs)

        # Only mark ops as exercised if the operation succeeded on MPS.
        if lowered and get_device_placement(result) == MPS_DEVICE:
            stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
            self.EXERCISED_STABLEHLO_OPS.update(
                _STABLEHLO_OP_RE.findall(stablehlo_text)
            )
        return result

    def evaluate_grad(self, argnum: int, jit: bool) -> tuple[jnp.ndarray]:
        """Evaluate the gradient of the operation. If the operation returns a tuple of
        values, gradients are evaluated for each element."""
        args = self.get_args()
        kwargs = self.get_kwargs()

        result = self.op(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            num_return_values = len(result)
        else:
            num_return_values = None

        grad_vals = []
        for returnnum in range(num_return_values or 1):

            def func(x):
                result = self.op(
                    *(x if i == argnum else arg for i, arg in enumerate(args)), **kwargs
                )
                if num_return_values is None:
                    assert isinstance(result, jnp.ndarray), (
                        f"Output of '{self.op}' is not a tensor."
                    )
                else:
                    result = result[returnnum]
                # Reduce to the mean if the output is not a scalar; we can only
                # differentiate scalars.
                if result.shape != ():
                    result = result.mean()
                return result

            lowered = None
            if jit:
                func = jax.jit(func)
                lowered = func.lower(args[argnum])
            grad_func = jax.grad(func)
            grad_vals.append(grad_func(args[argnum]))

            # Only mark ops as exercised if the operation succeeded on MPS.
            if lowered and get_device_placement(result) == MPS_DEVICE:
                stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
                self.EXERCISED_STABLEHLO_OPS.update(
                    _STABLEHLO_OP_RE.findall(stablehlo_text)
                )
        return tuple(grad_vals)


@pytest.fixture(autouse=True, scope="session")
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


def _make_unary_op_configs():
    return [
        OperationTestConfig(jnp.abs, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.ceil, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.cos, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.exp, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.flip, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.fliplr, numpy.random.normal(size=(17, 13))),
        OperationTestConfig(jnp.flipud, numpy.random.normal(size=(17, 13))),
        OperationTestConfig(jnp.floor, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.isfinite, numpy.asarray([0, jnp.nan, jnp.inf])),
        OperationTestConfig(jnp.log, numpy.random.gamma(5, size=(17,))),
        OperationTestConfig(jnp.log1p, numpy.random.gamma(5, size=(17,)) - 1),
        OperationTestConfig(jnp.negative, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.sign, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.sin, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.sqrt, numpy.random.gamma(5, size=(17,))),
        OperationTestConfig(jnp.tan, numpy.random.normal(size=(17,))),
        OperationTestConfig(jnp.tanh, numpy.random.normal(size=(17,))),
        OperationTestConfig(lax.rsqrt, numpy.random.gamma(5, size=(17,))),
        OperationTestConfig(
            special.erfinv,
            numpy.random.uniform(-0.9, 0.9, (17,)).astype(numpy.float32),
        ),
        OperationTestConfig(jnp.transpose, numpy.random.normal(size=(17, 8, 9))),
        OperationTestConfig(
            jnp.transpose,
            numpy.random.normal(size=(17, 8, 9)),
            (1, 0, 2),
            static_argnums=(1,),
        ),
    ]


def _make_binary_op_configs():
    return [
        OperationTestConfig(
            jnp.add, numpy.random.normal(size=(3, 4)), numpy.random.normal(size=(3, 1))
        ),
        OperationTestConfig(
            jnp.subtract,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 1)),
        ),
        OperationTestConfig(
            jnp.multiply,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 1)),
        ),
        OperationTestConfig(
            jnp.divide,
            numpy.random.normal(size=(3, 4)),
            numpy.random.gamma(5, size=(3, 1)),
        ),
        OperationTestConfig(
            jnp.dot,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(4, 5)),
        ),
        OperationTestConfig(
            jnp.less,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.less_equal,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.equal,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.greater_equal,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.greater,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.broadcast_arrays,
            numpy.random.normal(size=(3, 1)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.minimum,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.maximum,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.clip,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.clip,
            numpy.random.normal(size=(3, 4)),
            None,
            numpy.random.normal(size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.clip,
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(3, 4)),
            None,
        ),
        OperationTestConfig(
            jnp.power,
            numpy.random.normal(size=(5,)),
            numpy.random.gamma(5, size=(7, 1)),
        ),
        OperationTestConfig(
            jnp.power,
            numpy.random.gamma(5, size=(7, 1)),
            numpy.random.normal(size=(5,)),
        ),
        OperationTestConfig(
            lax.clamp,
            numpy.float32(-1.0),
            numpy.random.normal(size=(3, 4)),
            numpy.float32(1.0),
        ),
        OperationTestConfig(
            lax.rem,
            numpy.random.normal(size=(3, 4)),
            numpy.random.gamma(5, size=(3, 4)),
        ),
        OperationTestConfig(
            jnp.nextafter,
            numpy.array([1.0, -1.0, 0.0, 2.0], dtype=numpy.float32),
            numpy.array([2.0, -2.0, 1.0, 1.0], dtype=numpy.float32),
            differentiable_argnums=(),
        ),
    ]


def _make_random_op_configs():
    for shape in [(), (3,), (7, 8)]:
        yield from [
            OperationTestConfig(
                random.normal, random.key(17), shape, static_argnums=(1,)
            ),
            OperationTestConfig(
                random.truncated_normal,
                random.key(17),
                -0.1,
                0.2,
                shape,
                static_argnums=(3,),
            ),
            OperationTestConfig(random.split, random.key(18)),
            OperationTestConfig(random.split, random.key(18), 5, static_argnums=(1,)),
        ]


def _make_slice_op_configs():
    return [
        OperationTestConfig(
            lambda x, idx: x[idx],
            numpy.random.normal(size=(4, 5)),
            (numpy.random.randint(4), numpy.random.randint(5)),
        ),
        OperationTestConfig(
            lambda x, idx, y: x[idx],
            numpy.random.normal(size=(4, 5)),
            (numpy.random.randint(4), numpy.random.randint(5)),
            numpy.asarray(7.0),
        ),
        OperationTestConfig(
            lambda x: lax.dynamic_slice(x, (2,), (4,)),
            numpy.random.normal(size=(10,)),
        ),
    ]


def _make_shape_op_configs():
    return [
        OperationTestConfig(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            numpy.random.normal(size=(3, 4)),
            numpy.random.normal(size=(5, 4)),
        ),
        OperationTestConfig(
            lambda x: jnp.reshape(x, (20,)),
            numpy.random.normal(size=(4, 5)),
        ),
        OperationTestConfig(
            lambda x: jnp.pad(x, ((1, 1), (2, 2))),
            numpy.random.normal(size=(3, 3)),
            # Grad crashes with fatal Metal abort (sliceUpdateDataTensor shape mismatch).
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lambda x, idx: jnp.take(x, idx, axis=0),
            numpy.random.normal(size=(5, 3)),
            numpy.array([0, 2, 4]),
        ),
        OperationTestConfig(
            lambda x, idx, val: x.at[idx].set(val),
            numpy.random.normal(size=(5, 3)),
            numpy.array([0, 2]),
            numpy.random.normal(size=(2, 3)),
        ),
        OperationTestConfig(
            lambda x, update: lax.dynamic_update_slice(x, update, (1, 0)),
            numpy.random.normal(size=(5, 3)),
            numpy.random.normal(size=(2, 3)),
        ),
        OperationTestConfig(
            lambda x, idx, updates: x.at[idx].add(updates),
            numpy.zeros((10, 4), dtype=numpy.float32),
            numpy.array([0, 2, 5], dtype=numpy.int32),
            numpy.ones((3, 4), dtype=numpy.float32),
        ),
        OperationTestConfig(
            lambda x, idx, updates: x.at[idx].subtract(updates),
            numpy.ones((10, 4), dtype=numpy.float32),
            numpy.array([0, 2, 5], dtype=numpy.int32),
            numpy.full((3, 4), 0.1, dtype=numpy.float32),
        ),
        OperationTestConfig(
            lambda x, idx, updates: x.at[idx].mul(updates, unique_indices=True),
            numpy.ones((10, 4), dtype=numpy.float32),
            numpy.array([0, 2, 5], dtype=numpy.int32),
            numpy.full((3, 4), 2.0, dtype=numpy.float32),
        ),
        OperationTestConfig(
            lambda x, idx, updates: x.at[idx].divide(updates, unique_indices=True),
            numpy.ones((10, 4), dtype=numpy.float32),
            numpy.array([0, 2, 5], dtype=numpy.int32),
            numpy.full((3, 4), 2.0, dtype=numpy.float32),
        ),
        OperationTestConfig(
            lambda x, idx, updates: x.at[idx].power(updates, unique_indices=True),
            numpy.full((10, 4), 2.0, dtype=numpy.float32),
            numpy.array([0, 2, 5], dtype=numpy.int32),
            numpy.full((3, 4), 3.0, dtype=numpy.float32),
        ),
        OperationTestConfig(
            lambda x, idx, updates: x.at[idx].min(updates),
            numpy.random.normal(size=(10, 4)).astype(numpy.float32),
            numpy.array([0, 2, 5], dtype=numpy.int32),
            numpy.random.normal(size=(3, 4)).astype(numpy.float32),
        ),
        OperationTestConfig(
            lambda x, idx, updates: x.at[idx].max(updates),
            numpy.random.normal(size=(10, 4)).astype(numpy.float32),
            numpy.array([0, 2, 5], dtype=numpy.int32),
            numpy.random.normal(size=(3, 4)).astype(numpy.float32),
        ),
        OperationTestConfig(
            lambda x, kernel: lax.conv_general_dilated(
                x,
                kernel,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ),
            numpy.random.normal(size=(2, 8, 8, 3)).astype(numpy.float32),
            numpy.random.normal(size=(3, 3, 3, 8)).astype(numpy.float32),
        ),
        pytest.param(
            OperationTestConfig(
                lambda lhs, rhs: lax.conv(lhs, rhs, (1, 1), "SAME"),
                numpy.random.normal(size=(1, 3, 8, 8)),
                numpy.random.normal(size=(16, 3, 3, 3)),
            ),
            marks=pytest.mark.xfail(
                reason="conv backward pass uses unsupported kernel layout"
            ),
        ),
    ]


def _make_reduction_op_configs():
    return [
        OperationTestConfig(
            lambda x: jnp.sum(x),
            numpy.random.normal(size=(4, 5)),
        ),
        OperationTestConfig(
            lambda x: jnp.sum(x, axis=1),
            numpy.random.normal(size=(4, 5)),
        ),
        OperationTestConfig(
            lambda x: jnp.max(x, axis=0),
            numpy.random.normal(size=(4, 5)),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lambda x: jnp.min(x, axis=-1),
            numpy.random.normal(size=(4, 5)),
            differentiable_argnums=(),
        ),
    ]


def _make_conversion_op_configs():
    return [
        OperationTestConfig(
            lambda: jnp.arange(10, dtype=jnp.float32),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lambda x: x.astype(jnp.float16),
            numpy.random.normal(size=(4, 5)),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lambda x: lax.bitcast_convert_type(x, jnp.int32),
            numpy.random.normal(size=(4, 5)),
            differentiable_argnums=(),
        ),
    ]


def _make_special_op_configs():
    return [
        # This tests transfer of data with non-contiguous arrays.
        OperationTestConfig(
            lambda x: x,
            numpy.random.standard_normal((4, 5, 6, 8)).transpose((2, 0, 1, 3)),
        )
    ]


OPERATION_TEST_CONFIGS = [
    *_make_unary_op_configs(),
    *_make_binary_op_configs(),
    *_make_random_op_configs(),
    *_make_slice_op_configs(),
    *_make_shape_op_configs(),
    *_make_reduction_op_configs(),
    *_make_conversion_op_configs(),
    *_make_special_op_configs(),
]


@pytest.fixture(params=OPERATION_TEST_CONFIGS, ids=lambda op_config: op_config.op)
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
            try:
                result = op_config.evaluate_value(jit)
            except jax.errors.JaxRuntimeError as ex:
                if "Program contains unsupported StableHLO operations:" in str(ex):
                    pytest.skip(str(ex))
                raise
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
        pytest.skip(f"No differentiable arguments for operation '{op_config.op}'.")

    for argnum in argnums:
        results = []
        for platform in ["cpu", "mps"]:
            device = jax.devices(platform)[0]
            with jax.default_device(device):
                try:
                    result = op_config.evaluate_grad(argnum, jit)
                except jax.errors.JaxRuntimeError as ex:
                    if "Program contains unsupported StableHLO operations:" in str(ex):
                        pytest.skip(str(ex))
                    raise
                jax.tree.map_with_path(
                    lambda path, value: fassert(
                        value.device == device,
                        f"Value at '{path}' is on device {value.device}; expected {device}.",
                    ),
                    result,
                )
                results.append(result)

        jax.tree.map_with_path(assert_allclose_with_path, *results)
