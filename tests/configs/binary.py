import numpy
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig, complex_standard_normal


def make_binary_op_configs():
    for complex in [False, True]:
        with OperationTestConfig.module_name(
            "binary-complex" if complex else "binary-real"
        ):
            yield from [
                OperationTestConfig(
                    jnp.add,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 1), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.subtract,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 1), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.multiply,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 1), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.divide,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 1), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.dot,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (4, 5), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.less,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.less_equal,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.equal,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.greater_equal,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.greater,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.broadcast_arrays,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 1), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.minimum,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.maximum,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (3, 4), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.power,
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (5,), complex
                    ),
                    lambda rng: rng.gamma(5, size=(7, 1)),
                ),
                OperationTestConfig(
                    jnp.power,
                    lambda rng: rng.gamma(5, size=(7, 1)),
                    lambda rng, complex=complex: complex_standard_normal(
                        rng, (5,), complex
                    ),
                ),
            ]

        # Ops that do not support complex arguments, typically because they require an
        # order to be defined.
        yield from [
            OperationTestConfig(
                lax.rem,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.gamma(5, size=(3, 4)),
            ),
            OperationTestConfig(
                jnp.nextafter,
                numpy.array([1.0, -1.0, 0.0, 2.0], dtype=numpy.float32),
                numpy.array([2.0, -2.0, 1.0, 1.0], dtype=numpy.float32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.clip,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.clip,
                lambda rng: rng.standard_normal((3, 4)),
                None,
                lambda rng: rng.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.clip,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
                None,
            ),
            OperationTestConfig(
                lax.clamp,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.arctan2,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.bitwise_and,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.int32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.bitwise_xor,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.int32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.bitwise_or,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.int32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.bitwise_and,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.uint32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.uint32),
                differentiable_argnums=(),
                name="bitwise_and-uint32",
            ),
            OperationTestConfig(
                jnp.bitwise_xor,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.uint32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.uint32),
                differentiable_argnums=(),
                name="bitwise_xor-uint32",
            ),
            OperationTestConfig(
                jnp.bitwise_or,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.uint32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.uint32),
                differentiable_argnums=(),
                name="bitwise_or-uint32",
            ),
            OperationTestConfig(
                jnp.logical_and,
                numpy.array([True, False, True, False]),
                numpy.array([True, True, False, False]),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.logical_or,
                numpy.array([True, False, True, False]),
                numpy.array([True, True, False, False]),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.logical_xor,
                numpy.array([True, False, True, False]),
                numpy.array([True, True, False, False]),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lax.shift_left,
                numpy.array([1, 2, 4, 8], dtype=numpy.int32),
                numpy.array([0, 1, 2, 3], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.right_shift,
                numpy.array([-8, -1, 8, 127], dtype=numpy.int32),
                numpy.array([1, 31, 2, 40], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lax.shift_right_logical,
                numpy.array([1, 2, 4, 8, 0x80000000], dtype=numpy.uint32),
                numpy.array([0, 1, 2, 3, 31], dtype=numpy.uint32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lax.shift_left,
                numpy.array([0x40000000, -0x40000000, 5], dtype=numpy.int32),
                numpy.array([-2, -1, -31], dtype=numpy.int32),
                differentiable_argnums=(),
                name="lax.shift_left-negative-shift-count",
            ),
            OperationTestConfig(
                lax.shift_right_logical,
                numpy.array([0x40000000, -0x40000000, 5], dtype=numpy.int32),
                numpy.array([-2, -1, -31], dtype=numpy.int32),
                differentiable_argnums=(),
                name="lax.shift_right_logical-negative-shift-count",
            ),
            OperationTestConfig(
                lax.shift_right_arithmetic,
                numpy.array([0x40000000, -0x40000000, 5], dtype=numpy.int32),
                numpy.array([-2, -1, -31], dtype=numpy.int32),
                differentiable_argnums=(),
                name="lax.shift_right_arithmetic-negative-shift-count",
            ),
        ]
