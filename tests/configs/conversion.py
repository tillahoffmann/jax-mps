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
        ]
