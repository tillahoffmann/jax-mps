from jax import numpy as jnp
from jax import random

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
                        lambda key, complex=complex: complex_standard_normal(
                            key, (4, 8), complex
                        ),
                    ),
                    # Explicit argument because capture doesn't work.
                    OperationTestConfig(
                        lambda x, reduction=reduction: reduction(x, axis=1),
                        lambda key, complex=complex: complex_standard_normal(
                            key, (4, 8), complex
                        ),
                    ),
                ]

        with OperationTestConfig.module_name("reduction-real"):
            yield from [
                OperationTestConfig(
                    lambda x: jnp.max(x, axis=0),
                    lambda key: random.normal(key, (4, 8)),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.min(x, axis=-1),
                    lambda key: random.normal(key, (4, 8)),
                    differentiable_argnums=(),
                ),
            ]
