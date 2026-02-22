import jax
from jax import numpy as jnp
from jax import random

from .util import OperationTestConfig, complex_standard_normal


def make_shape_op_configs():
    # Flip and transpose ops for both real and complex inputs
    for complex in [False, True]:
        with OperationTestConfig.module_name(
            "shape-complex" if complex else "shape-real"
        ):
            yield from [
                OperationTestConfig(
                    jnp.flip,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (16,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.fliplr,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (8, 16), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.flipud,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (8, 16), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.transpose,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (4, 8, 16), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.transpose,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (4, 8, 16), complex
                    ),
                    (1, 0, 2),
                    static_argnums=(1,),
                ),
            ]

    with OperationTestConfig.module_name("shape"):
        yield from [
            OperationTestConfig(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                lambda key: random.normal(key, (3, 4)),
                lambda key: random.normal(key, (5, 4)),
            ),
            OperationTestConfig(
                lambda x: jnp.reshape(x, (20,)),
                lambda key: random.normal(key, (4, 5)),
            ),
            OperationTestConfig(
                lambda x: jnp.pad(x, ((1, 1), (2, 2))),
                lambda key: random.normal(key, (3, 3)),
                # Grad crashes with fatal Metal abort (sliceUpdateDataTensor shape mismatch).
                differentiable_argnums=(),
            ),
            # Pad with interior padding
            OperationTestConfig(
                lambda x: jax.lax.pad(x, 0.0, [(1, 1, 1), (0, 0, 2)]),
                lambda key: random.normal(key, (3, 4)),
                differentiable_argnums=(),
            ),
        ]
