import numpy
from jax import numpy as jnp

from .util import OperationTestConfig


def make_shape_op_configs():
    with OperationTestConfig.module_name("shape"):
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
        ]
