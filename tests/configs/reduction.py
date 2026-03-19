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
                ),
                OperationTestConfig(
                    lambda x: jnp.min(x, axis=-1),
                    lambda key: random.normal(key, (4, 8)),
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
            ),
            OperationTestConfig(
                lambda x: lax.cummax(x, axis=1),
                lambda key: random.normal(key, (3, 5)),
                name="cummax-axis1",
            ),
            OperationTestConfig(
                lambda x: lax.cummin(x, axis=1),
                lambda key: random.normal(key, (3, 5)),
                name="cummin-axis1",
            ),
        ]

    # Scalar reduce_window: identity operation on 0-dimensional input.
    with OperationTestConfig.module_name("reduction-real"):
        yield OperationTestConfig(
            lambda x: lax.reduce_window(x, 0.0, lax.add, (), (), "valid"),
            lambda key: random.normal(key, ()),
            name="reduce_window_scalar",
        )

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
            ),
            # Max pool 2D: window=(1,2,2,1), stride=(1,2,2,1) VALID
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "valid"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="maxpool2d-valid",
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
                # JAX VJP not implemented for window dilation
                differentiable_argnums=(),
            ),
            # Overlapping max pool: window=3, stride=1 (common in CNNs)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, -jnp.inf, lax.max, (1, 3, 3, 1), (1, 1, 1, 1), "valid"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="maxpool2d-overlapping",
            ),
            # Non-square window: 2x3 window
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, -jnp.inf, lax.max, (1, 2, 3, 1), (1, 2, 3, 1), "valid"
                ),
                lambda key: random.normal(key, (2, 8, 9, 3)),
                name="maxpool2d-nonsquare",
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
            ),
            # Min pool 1D: window=2, stride=2 on last axis (VALID padding)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, jnp.inf, lax.min, (1, 2), (1, 2), "valid"
                ),
                lambda key: random.normal(key, (2, 8)),
                name="minpool1d-valid",
            ),
            # Min pool 2D SAME padding: window=(1,3,3,1), stride=(1,1,1,1)
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, jnp.inf, lax.min, (1, 3, 3, 1), (1, 1, 1, 1), "same"
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                name="minpool2d-same",
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
                # JAX VJP not implemented for window dilation
                differentiable_argnums=(),
            ),
        ]

    # Tests targeting select_and_scatter edge cases (PR #100 regressions).
    with OperationTestConfig.module_name("reduction-real"):
        # Issue 1: Max pool gradient with tied values.
        # When multiple elements in a pooling window are equal (common after ReLU),
        # the gradient must match CPU semantics (first-occurrence wins), not
        # replicate the full gradient to every tied position.
        yield OperationTestConfig(
            lambda x: lax.reduce_window(
                x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "valid"
            ),
            # Deterministic input with many zeros (ties) — simulates post-ReLU.
            lambda key: jnp.array(
                [
                    [
                        [[0.0], [0.0], [1.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0]],
                        [[2.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [3.0]],
                    ]
                ],
                dtype=jnp.float32,
            ),
            name="maxpool2d-tied-grad",
        )

        # Issue 1b: Overlapping max pool with tied values (mask-based path).
        yield OperationTestConfig(
            lambda x: lax.reduce_window(
                x, -jnp.inf, lax.max, (1, 3, 3, 1), (1, 1, 1, 1), "valid"
            ),
            lambda key: jnp.array(
                [
                    [
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [5.0], [5.0], [0.0], [0.0]],
                        [[0.0], [5.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0], [0.0], [0.0]],
                    ]
                ],
                dtype=jnp.float32,
            ),
            name="maxpool2d-overlapping-tied-grad",
        )

        # Issue 1c: 1D max pool with tied values.
        yield OperationTestConfig(
            lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (1, 2), (1, 2), "valid"),
            lambda key: jnp.array(
                [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0]], dtype=jnp.float32
            ),
            name="maxpool1d-tied-grad",
        )

        # 1D max pool with SAME padding — tests 1D padding path.
        yield OperationTestConfig(
            lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (1, 3), (1, 2), "same"),
            lambda key: random.normal(key, (2, 7)),
            name="maxpool1d-same",
        )

        # Issue 1f: NCHW-style pooling layout — window on dims 2,3 with
        # batch=0 and channel=1. Tests the transpose logic in optimized paths.
        yield OperationTestConfig(
            lambda x: lax.reduce_window(
                x, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "valid"
            ),
            lambda key: random.normal(key, (2, 3, 8, 8)),
            name="maxpool2d-nchw-valid",
        )

        # Issue 2: 3D max pool (5D tensor) — tests that the generic fallback
        # still works for configurations outside the 1D/2D fast paths.
        yield OperationTestConfig(
            lambda x: lax.reduce_window(
                x,
                -jnp.inf,
                lax.max,
                (1, 2, 2, 2, 1),
                (1, 2, 2, 2, 1),
                "valid",
            ),
            lambda key: random.normal(key, (2, 4, 4, 4, 3)),
            name="maxpool3d-valid",
        )
