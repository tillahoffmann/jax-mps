import numpy
from jax import numpy as jnp

from .util import OperationTestConfig


def make_reduction_op_configs():
    with OperationTestConfig.module_name("reduction"):
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
