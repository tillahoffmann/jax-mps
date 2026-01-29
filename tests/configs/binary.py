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
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 1), complex),
                ),
                OperationTestConfig(
                    jnp.subtract,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 1), complex),
                ),
                OperationTestConfig(
                    jnp.multiply,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 1), complex),
                ),
                OperationTestConfig(
                    jnp.divide,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 1), complex),
                ),
                OperationTestConfig(
                    jnp.dot,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((4, 5), complex),
                ),
                OperationTestConfig(
                    jnp.less,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 4), complex),
                ),
                OperationTestConfig(
                    jnp.less_equal,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 4), complex),
                ),
                OperationTestConfig(
                    jnp.equal,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 4), complex),
                ),
                OperationTestConfig(
                    jnp.greater_equal,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 4), complex),
                ),
                OperationTestConfig(
                    jnp.greater,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 4), complex),
                ),
                OperationTestConfig(
                    jnp.broadcast_arrays,
                    complex_standard_normal((3, 1), complex),
                    complex_standard_normal((3, 4), complex),
                ),
                OperationTestConfig(
                    jnp.minimum,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 4), complex),
                ),
                OperationTestConfig(
                    jnp.maximum,
                    complex_standard_normal((3, 4), complex),
                    complex_standard_normal((3, 4), complex),
                ),
                OperationTestConfig(
                    jnp.power,
                    complex_standard_normal((5,), complex),
                    numpy.random.gamma(5, size=(7, 1)),
                ),
                OperationTestConfig(
                    jnp.power,
                    numpy.random.gamma(5, size=(7, 1)),
                    complex_standard_normal((5,), complex),
                ),
            ]

        # Ops that do not support complex arguments, typically because they require an
        # order to be defined.
        yield from [
            OperationTestConfig(
                lax.rem,
                numpy.random.standard_normal((3, 4)),
                numpy.random.gamma(5, size=(3, 4)),
            ),
            OperationTestConfig(
                jnp.nextafter,
                numpy.array([1.0, -1.0, 0.0, 2.0], dtype=numpy.float32),
                numpy.array([2.0, -2.0, 1.0, 1.0], dtype=numpy.float32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.clip,
                numpy.random.standard_normal((3, 4)),
                numpy.random.standard_normal((3, 4)),
                numpy.random.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.clip,
                numpy.random.standard_normal((3, 4)),
                None,
                numpy.random.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.clip,
                numpy.random.standard_normal((3, 4)),
                numpy.random.standard_normal((3, 4)),
                None,
            ),
            OperationTestConfig(
                lax.clamp,
                numpy.random.standard_normal((3, 4)),
                numpy.random.standard_normal((3, 4)),
                numpy.random.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.arctan2,
                numpy.random.standard_normal((3, 4)),
                numpy.random.standard_normal((3, 4)),
            ),
        ]
