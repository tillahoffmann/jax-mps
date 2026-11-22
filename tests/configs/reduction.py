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
            # 1-element axis: cumulative detection skips axes where
            # windowDims[i]==1, so this falls through to the pooling tier
            # — which must accept Prod for the lowering to succeed.
            OperationTestConfig(
                lambda x: jnp.cumprod(x, axis=0),
                lambda key: random.uniform(key, (1,), minval=0.5, maxval=1.5),
                name="cumprod-axis0-size1",
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
            # Prod pool 2D: same pattern as sum, but with mul reduction. The
            # JAX upstream tests (testCumSumProd, testSphHarmY) lower
            # cumulative-prod to reduce_window when the cumulative pattern
            # isn't detected, so the pooling tier must support Prod too.
            # JAX itself doesn't define a reverse-mode VJP for prod
            # reduce_window, so this config exercises the forward path only.
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, 1.0, lax.mul, (1, 2, 2, 1), (1, 2, 2, 1), "valid"
                ),
                lambda key: random.uniform(key, (2, 8, 8, 3), minval=0.5, maxval=1.5),
                name="prodpool2d-valid",
                differentiable_argnums=(),
            ),
            # Prod pool with SAME padding — exercises the padded prod branch.
            # MLX's pad fills boundary slots with the init value (1, the
            # multiplicative identity) so out-of-bounds slots don't change
            # the product. Required to confirm the validation logic accepts
            # this path.
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, 1.0, lax.mul, (1, 3, 3, 1), (1, 1, 1, 1), "same"
                ),
                lambda key: random.uniform(key, (2, 8, 8, 3), minval=0.5, maxval=1.5),
                name="prodpool2d-same",
                differentiable_argnums=(),
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
            # Base dilation 1D: dilates the *input* by inserting init-valued
            # holes between elements before windowing. This is what sum-pool
            # gradients lower to (base_dilation == forward stride).
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x, 0.0, lax.add, (1, 2), (1, 1), "valid", (1, 2), None
                ),
                lambda key: random.normal(key, (2, 5)),
                name="sumpool1d-base-dilation",
                # JAX VJP not implemented for base dilation; forward only.
                differentiable_argnums=(),
            ),
            # Base dilation 2D with sum reduction and SAME padding.
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x,
                    0.0,
                    lax.add,
                    (1, 2, 2, 1),
                    (1, 1, 1, 1),
                    "same",
                    (1, 2, 2, 1),
                    None,
                ),
                lambda key: random.normal(key, (2, 6, 6, 3)),
                name="sumpool2d-base-dilation-same",
                differentiable_argnums=(),
            ),
            # Base dilation with max reduction (holes filled with init=-inf so
            # they never win the window).
            OperationTestConfig(
                lambda x: lax.reduce_window(
                    x,
                    -jnp.inf,
                    lax.max,
                    (1, 2, 2, 1),
                    (1, 1, 1, 1),
                    "valid",
                    (1, 2, 2, 1),
                    None,
                ),
                lambda key: random.normal(key, (2, 6, 6, 3)),
                name="maxpool2d-base-dilation",
                differentiable_argnums=(),
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
                lambda x: (
                    lax.reduce_window(
                        x, 0.0, lax.add, (1, 2, 2, 1), (1, 2, 2, 1), "valid"
                    )
                    / 4.0
                ),
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

    # Bitwise reductions (stablehlo.reduce with and/or/xor bodies). MLX has no
    # native bitwise reduce for integers, so these exercise the fold fallback;
    # bool inputs exercise the all/any/parity fast paths. Non-identity
    # initializers and empty reduced dimensions must honor the init value.
    #
    # Inputs are biased so the fold over the input does not saturate (AND
    # towards all-zero, OR towards all-one); otherwise a dropped initializer
    # would go unnoticed. Integer inputs keep bits 20-30 fixed (set for AND,
    # clear for OR/XOR) so the initializer's high bits always show up.
    def _bool_input(shape, p):
        return lambda key: random.bernoulli(key, p, shape)

    def _int_input(shape, high_bits=0):
        return lambda key: (
            random.randint(key, shape, 0, 1 << 20, dtype=jnp.int32) | high_bits
        )

    bitwise_init = 0x0A5A5A5A

    with OperationTestConfig.module_name("reduction-bitwise"):
        for name, op in [
            ("and", lax.bitwise_and),
            ("or", lax.bitwise_or),
            ("xor", lax.bitwise_xor),
        ]:
            identity_int = -1 if name == "and" else 0
            identity_bool = name == "and"
            high_bits = 0x7FF00000 if name == "and" else 0
            p = {"and": 0.9, "or": 0.1, "xor": 0.5}[name]
            for shape, axes, suffix in [
                ((4, 7), (1,), "axis1"),
                ((3, 5, 6), (0, 2), "axes02"),
                ((3, 0), (1,), "empty"),
                ((1,), (0,), "size1"),
            ]:
                yield OperationTestConfig(
                    lambda x, op=op, axes=axes, init=identity_int: lax.reduce(
                        x, jnp.int32(init), op, axes
                    ),
                    _int_input(shape, high_bits),
                    name=f"reduce_{name}-int-{suffix}",
                )
                yield OperationTestConfig(
                    lambda x, op=op, axes=axes, init=identity_bool: lax.reduce(
                        x, jnp.bool_(init), op, axes
                    ),
                    _bool_input(shape, p),
                    name=f"reduce_{name}-bool-{suffix}",
                )
                # Non-identity initializer: must be folded in exactly once.
                # StableHLO leaves non-identity inits implementation-defined,
                # and XLA CPU (the reference) drops the init when the reduced
                # size is 1 (it rewrites the reduce to a reshape), so there is
                # no stable reference for that case.
                if suffix == "size1":
                    continue
                yield OperationTestConfig(
                    lambda x, op=op, axes=axes: lax.reduce(
                        x, jnp.int32(bitwise_init), op, axes
                    ),
                    _int_input(shape, high_bits),
                    name=f"reduce_{name}-int-init-{suffix}",
                )
                yield OperationTestConfig(
                    lambda x, op=op, axes=axes, init=not identity_bool: lax.reduce(
                        x, jnp.bool_(init), op, axes
                    ),
                    _bool_input(shape, p),
                    name=f"reduce_{name}-bool-init-{suffix}",
                )
        # Long axis: exercises the logarithmic tree fold with odd tails.
        yield OperationTestConfig(
            lambda x: lax.reduce(x, jnp.int32(0), lax.bitwise_xor, (0,)),
            _int_input((1023, 3)),
            name="reduce_xor-int-long",
        )

    # Non-identity initializers for the MLX-native reductions (sum/prod/max/min).
    # StableHLO folds the init into every reduction (and every window) exactly
    # once; dropping it silently returns the identity-init answer. Values are
    # chosen to land inside the data range so a dropped or doubled init shows.
    # Reduced sizes and windows are > 1 because XLA CPU (the reference) drops
    # the init for size-1 reductions. JAX has no VJP for reductions with a
    # non-monoid init, so these are forward-only.
    def _uniform(shape, lo, hi):
        return lambda key: random.uniform(key, shape, minval=lo, maxval=hi)

    init_cases = [
        ("sum", lax.add, 2.5, _uniform((4, 7), -1.0, 1.0)),
        ("prod", lax.mul, 2.0, _uniform((4, 7), 0.5, 1.5)),
        ("max", lax.max, 0.5, _uniform((4, 7), -1.0, 1.0)),
        ("min", lax.min, -0.5, _uniform((4, 7), -1.0, 1.0)),
    ]
    with OperationTestConfig.module_name("reduction-init"):
        for name, op, init, make_input in init_cases:
            for axes, suffix in [((1,), "axis1"), ((0, 1), "all")]:
                yield OperationTestConfig(
                    lambda x, op=op, init=init, axes=axes: lax.reduce(
                        x, jnp.float32(init), op, axes
                    ),
                    make_input,
                    name=f"reduce_{name}-init-{suffix}",
                    differentiable_argnums=(),
                )
            for padding in ["valid", "same"]:
                yield OperationTestConfig(
                    lambda x, op=op, init=init, padding=padding: lax.reduce_window(
                        x, jnp.float32(init), op, (2, 3), (1, 2), padding
                    ),
                    make_input,
                    name=f"reduce_window_{name}-init-{padding}",
                    differentiable_argnums=(),
                )
            # Full-length, low-padded window along one axis: the cumulative
            # (cumsum/cumprod/cummax/cummin) tier.
            yield OperationTestConfig(
                lambda x, op=op, init=init: lax.reduce_window(
                    x, jnp.float32(init), op, (1, 7), (1, 1), ((0, 0), (6, 0))
                ),
                make_input,
                name=f"reduce_window_{name}-init-cumulative",
                differentiable_argnums=(),
            )
        # Base-dilation holes, like padding, must not pick up the init.
        yield OperationTestConfig(
            lambda x: lax.reduce_window(
                x, jnp.float32(2.5), lax.add, (2, 3), (1, 1), "valid", (1, 2)
            ),
            _uniform((4, 7), -1.0, 1.0),
            name="reduce_window_sum-init-base-dilation",
            differentiable_argnums=(),
        )
        yield OperationTestConfig(
            lambda x: lax.reduce(x, jnp.int32(7), lax.add, (1,)),
            lambda key: random.randint(key, (4, 7), -50, 50, dtype=jnp.int32),
            name="reduce_sum-int-init",
        )
        yield OperationTestConfig(
            lambda x: lax.reduce(x, jnp.int32(40), lax.max, (1,)),
            lambda key: random.randint(key, (4, 7), -50, 50, dtype=jnp.int32),
            name="reduce_max-int-init",
        )
