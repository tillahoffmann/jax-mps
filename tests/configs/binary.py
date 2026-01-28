import numpy
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig


def make_binary_op_configs():
    with OperationTestConfig.module_name("binary"):
        return [
            OperationTestConfig(
                jnp.add,
                numpy.random.normal(size=(3, 4)),
                numpy.random.normal(size=(3, 1)),
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
