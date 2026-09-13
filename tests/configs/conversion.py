import jax
import numpy
from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def make_conversion_op_configs():
    with OperationTestConfig.module_name("conversion"):
        return [
            OperationTestConfig(
                lambda: jnp.arange(10, dtype=jnp.float32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x: x.astype(jnp.float16),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
            ),
            # Use deterministic input for bitcast tests since random.normal can
            # produce slightly different values on MPS vs CPU, which becomes
            # visible when bitcasting to integers.
            OperationTestConfig(
                lambda x: lax.bitcast_convert_type(x, jnp.int32),
                numpy.arange(32, dtype=numpy.float32).reshape(4, 8),
                differentiable_argnums=(),
            ),
            # jax-mps#231: MLX has no float64, so f64 constants are narrowed to
            # float32. That used to change only the dtype tag, leaving the
            # 8-byte payload to be read through a `const float*` (pi decoded as
            # 3.37e12). x64 is required to make JAX emit an f64 constant at all;
            # the inputs stay float32 so the buffer transfer still succeeds.
            OperationTestConfig(
                # jax.jit inside the lambda: the harness also runs configs in
                # eager mode, where an f64 scalar would be transferred as an f64
                # *buffer* (correctly rejected) instead of embedded as a
                # constant. Jitting here keeps it a constant in both modes.
                # The astype keeps the *output* f32 so the eager grad's cotangent
                # is not an f64 buffer either; the f64 constant still lowers.
                jax.jit(lambda x: (x + numpy.float64(numpy.pi)).astype(jnp.float32)),
                numpy.zeros(3, numpy.float32),
                config_overrides={"jax_enable_x64": True},
                name="f64_constant_splat",
            ),
            OperationTestConfig(
                jax.jit(
                    lambda x: (x + numpy.array([1.5, 2.25, 3.125])).astype(jnp.float32)
                ),
                numpy.zeros(3, numpy.float32),
                config_overrides={"jax_enable_x64": True},
                name="f64_constant_dense",
            ),
            # Narrowing must run before the jax-mps#170 non-finite bitcast, so
            # that path still sees an inf rather than a large finite float.
            OperationTestConfig(
                jax.jit(
                    lambda x: jnp.minimum(x, numpy.float64(numpy.inf)).astype(
                        jnp.float32
                    )
                ),
                # Nonzero: inf's f64 bit pattern truncates to 0.0f, and
                # min(0, 0) == min(0, inf) would pass on a zero input.
                numpy.array([1.0, 2.0, 3.0], numpy.float32),
                config_overrides={"jax_enable_x64": True},
                name="f64_constant_inf",
            ),
            # complex<f64> takes the same narrowing path (16 bytes -> 8).
            # Distinct real and imaginary parts catch a component swap.
            OperationTestConfig(
                jax.jit(
                    lambda x: (x + numpy.complex128(2.5 - 7.25j)).astype(jnp.complex64)
                ),
                numpy.zeros(3, numpy.complex64),
                config_overrides={"jax_enable_x64": True},
                name="c128_constant_splat",
            ),
        ]
