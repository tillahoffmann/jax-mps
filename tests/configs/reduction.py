from jax import lax, random
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

    # Cumulative operations (lower to stablehlo.reduce_window).
    with OperationTestConfig.module_name("reduction-real"):
        yield from [
            OperationTestConfig(
                lambda x: jnp.cumsum(x, axis=1),
                lambda key: random.normal(key, (3, 5)),
                name="cumsum-axis1",
            ),
            OperationTestConfig(
                lambda x: jnp.cumsum(x, axis=0),
                lambda key: random.normal(key, (4, 6)),
                name="cumsum-axis0",
            ),
            OperationTestConfig(
                lambda x: jnp.cumprod(x, axis=1),
                lambda key: random.uniform(key, (3, 5), minval=0.5, maxval=1.5),
                name="cumprod-axis1",
                differentiable_argnums=(),  # grad uses pad with interior dilation
            ),
            OperationTestConfig(
                lambda x: lax.cummax(x, axis=1),
                lambda key: random.normal(key, (3, 5)),
                name="cummax-axis1",
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x: lax.cummin(x, axis=1),
                lambda key: random.normal(key, (3, 5)),
                name="cummin-axis1",
                differentiable_argnums=(),
            ),
        ]

    # Pooling operations (lower to stablehlo.reduce_window with pooling pattern).
    with OperationTestConfig.module_name("reduction-real"):
        yield from [
            # Max pool 1D: window=2, stride=2 on last axis (VALID padding)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, -jnp.inf, lax.max, (1, 2), (1, 2), "valid"
                ),
                lambda key: random.normal(key, (2, 8)),
                name="maxpool1d-valid",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
            # Max pool 2D: window=(1,2,2,1), stride=(1,2,2,1) VALID
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "valid"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="maxpool2d-valid",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
            # Sum pool 2D: window=(1,2,2,1), stride=(1,2,2,1) VALID
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, 0.0, lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "valid"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="sumpool2d-valid",
            ),
            # Max pool 2D SAME padding: window=(1,3,3,1), stride=(1,1,1,1)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, -jnp.inf, lax.max, (1, 3, 3, 1), (1, 1, 1, 1), "same"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="maxpool2d-same",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
            # Max pool 2D with window dilation: window=(1,2,2,1), stride=(1,1,1,1),
            # window_dilation=(1,2,2,1) and VALID padding.
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x,
                    -jnp.inf,
                    lax.max,
                    (1, 2, 2, 1),
                    (1, 1, 1, 1),
                    "valid",
                    None,
                    (1, 2, 2, 1),
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="maxpool2d-window-dilation",
                # Grad requires pad with interior/window dilation (not yet supported)
                differentiable_argnums=(),
            ),
            # Overlapping max pool: window=3, stride=1 (common in CNNs)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, -jnp.inf, lax.max, (1, 3, 3, 1), (1, 1, 1, 1), "valid"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="maxpool2d-overlapping",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
            # Non-square window: 2x3 window
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, -jnp.inf, lax.max, (1, 2, 3, 1), (1, 2, 3, 1), "valid"
                ),
                lambda key: random.normal(key, (2, 8, 9, 3)),
                name="maxpool2d-nonsquare",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
            # Sum pool with SAME padding
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, 0.0, lax.add, (1, 3, 3, 1), (1, 1, 1, 1), "same"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="sumpool2d-same",
            ),
            # Average pool 2D (sum pool divided by window area)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, 0.0, lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "valid"
                )
                / 4.0,
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="avgpool2d-valid",
            ),
            # Min pool 2D: window=(1,2,2,1), stride=(1,2,2,1) VALID
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, jnp.inf, lax.min, (1, 2, 2, 1), (1, 2, 2, 1), "valid"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="minpool2d-valid",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
            # Min pool 1D: window=2, stride=2 on last axis (VALID padding)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, jnp.inf, lax.min, (1, 2), (1, 2), "valid"
                ),
                lambda key: random.normal(key, (2, 8)),
                name="minpool1d-valid",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
            # Min pool 2D SAME padding: window=(1,3,3,1), stride=(1,1,1,1)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, jnp.inf, lax.min, (1, 3, 3, 1), (1, 1, 1, 1), "same"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="minpool2d-same",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
            # Min pool 2D with window dilation
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x,
                    jnp.inf,
                    lax.min,
                    (1, 2, 2, 1),
                    (1, 1, 1, 1),
                    "valid",
                    None,
                    (1, 2, 2, 1),
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="minpool2d-window-dilation",
                # Grad requires select_and_scatter (not yet supported)
                differentiable_argnums=(),
            ),
        ]
