import re
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Sequence

import jax
import numpy
import pytest
from jax import numpy as jnp

CPU_DEVICE = jax.devices("cpu")[0]
MPS_DEVICE = (
    jax.devices("mps")[0] if "mps" in {d.platform for d in jax.devices()} else None
)
STABLEHLO_OP_RE = re.compile(r"(?<![\#\!])(?:stablehlo|chlo)\.[\w\.]+")


def xfail_match(pattern: str) -> pytest.MarkDecorator:
    """Create a strict xfail marker that validates the error message pattern."""
    return pytest.mark.xfail(reason=pattern, match=pattern, strict=True)  # pyright: ignore[reportCallIssue]


def get_device_placement(value):
    """Get the device placement of a PyTree."""
    device = None
    for leaf in jax.tree.flatten(value)[0]:
        if not isinstance(leaf, jax.Array):
            continue
        assert device is None or device == leaf.device, "Mixed device placement."
        device = leaf.device
    assert device is not None, "Failed to infer device placement."
    return device


def complex_standard_normal(
    rng: numpy.random.Generator, shape: tuple[int, ...], complex: bool
) -> numpy.ndarray:
    """Generate random normal data, optionally complex-valued."""
    if complex:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    else:
        return rng.standard_normal(shape)


class OperationTestConfig:
    """Configuration for testing operations.

    Args:
        func: Function to test.
        *args: Factory functions for positional arguments. Non-callables will
            automatically be wrapped in lambdas.
        **kwargs: Factory functions for keyword arguments. Non-callables will
            automatically be wrapped in lambdas.
        differentiable_argnums: Position of arguments that can be differentiated with
            respect to. Defaults to positional arguments with inexact types.
        static_argnums: Position of arguments that should be treated as static in
            jit-compile.
        name: Display name of the operation test config.
    """

    EXERCISED_STABLEHLO_OPS: set[str] = {
        # HACK: Register these ops as exercised because JAX doesn't seem to generate them.
        "stablehlo.broadcast",
        "stablehlo.dot",
        "stablehlo.erf",
    }
    ACTIVE_MODULE_NAME: ClassVar[str | None] = None

    @classmethod
    @contextmanager
    def module_name(cls, name: str):
        assert cls.ACTIVE_MODULE_NAME is None, (
            f"Module name '{cls.ACTIVE_MODULE_NAME}' is already active."
        )
        cls.ACTIVE_MODULE_NAME = name
        yield
        cls.ACTIVE_MODULE_NAME = None

    def __init__(
        self,
        func: Callable,
        *args: Any,
        differentiable_argnums: Sequence[int] | None = None,
        static_argnums: Sequence[int] | None = None,
        grad_transform: Callable | None = None,
        name: str | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.func = func
        self.differentiable_argnums = differentiable_argnums
        self.static_argnums = static_argnums
        self.grad_transform = grad_transform or jax.grad
        self.seed = seed
        # Wrap non-callables in lambdas that accept (and ignore) rng
        self.args = [
            arg if callable(arg) else (lambda rng, arg=arg: arg) for arg in args
        ]
        self.kwargs = {
            key: arg if callable(arg) else (lambda rng, arg=arg: arg)
            for key, arg in kwargs.items()
        }

        if name is None:
            name = self.func.__name__
        if self.ACTIVE_MODULE_NAME:
            name = f"{self.ACTIVE_MODULE_NAME}.{name}"
        self.name = name

    def get_args(self, rng: numpy.random.Generator):
        """Get positional arguments, using rng for any random generation."""
        args = []
        for arg_func in self.args:
            arg = arg_func(rng)
            if isinstance(arg, numpy.ndarray):
                arg = jnp.asarray(arg)
            args.append(arg)
        return args

    def get_kwargs(self, rng: numpy.random.Generator):
        """Get keyword arguments, using rng for any random generation."""
        return {key: arg_func(rng) for key, arg_func in self.kwargs.items()}

    def get_differentiable_argnums(self) -> tuple[int, ...]:
        """Get a tuple of integers indicating which arguments can be differentiated with
        respect to."""
        if self.differentiable_argnums is not None:
            return tuple(self.differentiable_argnums)

        rng = numpy.random.default_rng(self.seed)
        differentiable_argnums: list[int] = []
        for argnum, arg in enumerate(self.get_args(rng)):
            if isinstance(arg, float):
                differentiable_argnums.append(argnum)
            elif isinstance(arg, jnp.ndarray):
                if arg.dtype == jnp.float32 or arg.dtype == jnp.complex64:
                    differentiable_argnums.append(argnum)
        return tuple(differentiable_argnums)

    def evaluate_value(self, jit: bool):
        """Evaluate the output of the operation."""
        rng = numpy.random.default_rng(self.seed)
        args = self.get_args(rng)
        kwargs = self.get_kwargs(rng)
        lowered = None
        func = self.func
        if jit:
            func = jax.jit(func, static_argnums=self.static_argnums)
            lowered = func.lower(*args, **kwargs)
        result = func(*args, **kwargs)

        # Only mark ops as exercised if the operation succeeded on MPS.
        if lowered and get_device_placement(result) == MPS_DEVICE:
            stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
            self.EXERCISED_STABLEHLO_OPS.update(STABLEHLO_OP_RE.findall(stablehlo_text))
        return result

    def evaluate_grad(self, argnum: int, jit: bool) -> tuple[jnp.ndarray]:
        """Evaluate the gradient of the operation. If the operation returns a tuple of
        values, gradients are evaluated for each element."""
        rng = numpy.random.default_rng(self.seed)
        args = self.get_args(rng)
        kwargs = self.get_kwargs(rng)

        func = self.func
        result = func(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            num_return_values = len(result)
        else:
            num_return_values = None

        grad_vals = []
        for returnnum in range(num_return_values or 1):

            def target(*args, **kwargs):
                result = func(*args, **kwargs)
                if num_return_values is None:
                    assert isinstance(result, jnp.ndarray), (
                        f"Output of '{func}' is not a tensor."
                    )
                else:
                    result = result[returnnum]

                # FIXME: Reduce to the magnitude if the function is complex-valued so we
                # don't have to deal with complex derivatives. This isn't ideal but
                # pragmatic.
                if result.dtype == jnp.complex64:
                    result = jnp.abs(result)

                # Reduce to the mean if the output is not a scalar; we can only
                # differentiate scalars.
                if result.shape != ():
                    result = result.mean()
                return result

            grad_func = self.grad_transform(target, argnums=argnum)
            lowered = None
            if jit:
                grad_func = jax.jit(grad_func, static_argnums=self.static_argnums)
                lowered = grad_func.lower(*args)
            grad_vals.append(grad_func(*args))

            # Only mark ops as exercised if the operation succeeded on MPS.
            if lowered and get_device_placement(result) == MPS_DEVICE:
                stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
                self.EXERCISED_STABLEHLO_OPS.update(
                    STABLEHLO_OP_RE.findall(stablehlo_text)
                )
        return tuple(grad_vals)
