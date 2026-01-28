import numpy
from jax import lax
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
                numpy.random.normal(size=(4, 5)),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x: lax.bitcast_convert_type(x, jnp.int32),
                numpy.random.normal(size=(4, 5)),
                differentiable_argnums=(),
            ),
        ]
