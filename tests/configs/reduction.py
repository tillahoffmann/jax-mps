import numpy
from jax import numpy as jnp

from .util import OperationTestConfig, complex_standard_normal


def make_reduction_op_configs():
    for complex in [True, False]:
        with OperationTestConfig.module_name(
            "reduction-complex" if complex else "reduction-real"
        ):
            for reduction in [jnp.sum, jnp.mean, jnp.var, jnp.std]:
                yield from [
                    OperationTestConfig(
                        reduction,
                        complex_standard_normal((4, 5), complex),
                    ),
                    # Explicit argument because capture doesn't work.
                    OperationTestConfig(
                        lambda x, reduction=reduction: reduction(x, axis=1),
                        complex_standard_normal((4, 5), complex),
                    ),
                ]

        with OperationTestConfig.module_name("reduction-real"):
            yield from [
                OperationTestConfig(
                    lambda x: jnp.max(x, axis=0),
                    numpy.random.standard_normal((4, 5)),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.min(x, axis=-1),
                    numpy.random.standard_normal((4, 5)),
                    differentiable_argnums=(),
                ),
            ]
