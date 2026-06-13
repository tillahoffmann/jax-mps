"""Configs for tests/test_fusion.py.

A FusionTestConfig is like OperationTestConfig but focused on IR-level
fusion assertions: it carries a JAX function + argument factories, plus
expected `@mps.*` custom_calls that the plugin's fusion passes should
produce in the post-pass StableHLO.

The actual test harness (tests/test_fusion.py) lowers and runs each
function on MPS with JAX_MPS_DUMP_OPTIMIZED_IR set, then inspects the
dumped module files for the expected ops and also asserts numerical
equivalence to the reference (CPU) output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jax
import jax.nn as jnn
import numpy
from jax import numpy as jnp
from jax import random


@dataclass
class FusionTestConfig:
    """One fusion test case.

    Attributes:
        name: Display name (becomes the pytest id).
        func: The JAX function under test.
        args: Factory functions producing positional args (same shape as
            OperationTestConfig — non-callables get wrapped in lambdas).
        expected_custom_calls: dict mapping `@mps.xxx` target names to the
            minimum count that must appear in the post-pass IR. e.g.
            ``{"mps.addmm": 1}``. Use 0 to assert absence.
        atol / rtol: tolerance vs. the CPU reference (cross-hardware —
            stays loose to absorb reduction-order drift).
        fusion_atol / fusion_rtol: tolerance for fused-MPS vs unfused-MPS
            on the same hardware. Tighter: a regression that's specific to
            the fusion itself should show up here.
    """

    name: str
    func: Callable[..., Any]
    args: Sequence[Any] = field(default_factory=tuple)
    expected_custom_calls: dict[str, int] = field(default_factory=dict)
    atol: float = 1e-5
    rtol: float = 1e-5
    fusion_atol: float = 1e-6
    fusion_rtol: float = 1e-6
    seed: int = 0

    def __post_init__(self):
        self.args = tuple(a if callable(a) else (lambda key, a=a: a) for a in self.args)

    def make_args(self) -> tuple[Any, ...]:
        key = random.key(self.seed)
        out = []
        for factory in self.args:
            key, sub = random.split(key)
            val = factory(sub)
            if isinstance(val, numpy.ndarray):
                val = jnp.asarray(val)
            out.append(val)
        return tuple(out)


def make_fusion_configs() -> list[FusionTestConfig]:
    """Enumerate the fusion cases. Add new cases here as fusions land."""
    configs: list[FusionTestConfig] = []

    def randn(*shape, key_mul=1):
        def f(key):
            return random.normal(key, shape) * key_mul

        return f

    # --- fuse_bias_add: stablehlo.dot_general + broadcast + add -> mps.addmm ---

    # Simple 2D Linear(bias=True).
    configs.append(
        FusionTestConfig(
            name="addmm.2d",
            func=lambda x, w, b: x @ w + b,
            args=(randn(16, 32), randn(32, 8), randn(8)),
            expected_custom_calls={"mps.addmm": 1},
        )
    )

    # 3D Linear(bias=True) (batched contraction — the transformer-Linear shape).
    configs.append(
        FusionTestConfig(
            name="addmm.3d",
            func=lambda x, w, b: x @ w + b,
            args=(randn(4, 16, 32), randn(32, 8), randn(8)),
            expected_custom_calls={"mps.addmm": 1},
        )
    )

    # Two chained Linears — both should fuse.
    configs.append(
        FusionTestConfig(
            name="addmm.chained",
            func=lambda x, w1, b1, w2, b2: (x @ w1 + b1) @ w2 + b2,
            args=(randn(16, 32), randn(32, 64), randn(64), randn(64, 8), randn(8)),
            expected_custom_calls={"mps.addmm": 2},
        )
    )

    # Add + broadcast where the right operand is NOT a 1-D bias — must NOT fuse.
    configs.append(
        FusionTestConfig(
            name="addmm.non_bias_add",
            func=lambda x, w, other: x @ w + other,
            args=(randn(16, 32), randn(32, 8), randn(16, 8)),
            expected_custom_calls={"mps.addmm": 0},
        )
    )

    # Scalar bias (0-D): must NOT fuse (no trailing-dim bias).
    configs.append(
        FusionTestConfig(
            name="addmm.scalar_bias",
            func=lambda x, w, s: x @ w + s,
            args=(randn(16, 32), randn(32, 8), jnp.float32(0.5)),
            expected_custom_calls={"mps.addmm": 0},
        )
    )

    # Batched matmul with leading batch dims + bias on trailing output dim.
    # Exercises the batching-dim branch of hasStandardMatmulLayout.
    configs.append(
        FusionTestConfig(
            name="addmm.batched",
            func=lambda x, w, b: jnp.einsum("bij,bjk->bik", x, w) + b,
            args=(randn(4, 16, 32), randn(4, 32, 8), randn(8)),
            expected_custom_calls={"mps.addmm": 1},
        )
    )

    # dot_general with multiple free dims on rhs: `(M, K) @ (K, N1, N2) -> (M, N1, N2)`.
    # Not a standard (batched) matmul; addmm can't represent this, so must NOT fuse.
    configs.append(
        FusionTestConfig(
            name="addmm.rhs_multi_free_dims",
            func=lambda x, w, b: jnp.einsum("mk,knp->mnp", x, w) + b,
            args=(randn(16, 32), randn(32, 8, 4), randn(4)),
            expected_custom_calls={"mps.addmm": 0},
        )
    )

    # --- fuse_softmax: reduce_max/sub/exp/reduce_sum/divide -> mps.softmax ---

    # Trailing-axis softmax at several ranks & shapes.
    for shape in [(8,), (4, 16), (2, 3, 12), (2, 4, 8, 32)]:
        configs.append(
            FusionTestConfig(
                name=f"softmax.trailing.{'x'.join(map(str, shape))}",
                func=lambda x: jnn.softmax(x, axis=-1),
                args=(randn(*shape),),
                expected_custom_calls={"mps.softmax": 1},
            )
        )

    # Non-trailing axis (explicit).
    configs.append(
        FusionTestConfig(
            name="softmax.axis0",
            func=lambda x: jnn.softmax(x, axis=0),
            args=(randn(16, 32),),
            expected_custom_calls={"mps.softmax": 1},
        )
    )
    configs.append(
        FusionTestConfig(
            name="softmax.axis1_of_3",
            func=lambda x: jnn.softmax(x, axis=1),
            args=(randn(4, 16, 8),),
            expected_custom_calls={"mps.softmax": 1},
        )
    )
    # Negative axis (should normalize to positive inside the pass).
    configs.append(
        FusionTestConfig(
            name="softmax.negative_axis",
            func=lambda x: jnn.softmax(x, axis=-2),
            args=(randn(4, 8, 16),),
            expected_custom_calls={"mps.softmax": 1},
        )
    )
    # Larger batch dim — shape-invariance sanity.
    configs.append(
        FusionTestConfig(
            name="softmax.big_batch",
            func=lambda x: jnn.softmax(x, axis=-1),
            args=(randn(64, 128),),
            expected_custom_calls={"mps.softmax": 1},
        )
    )
    # Two independent softmaxes — both should fuse.
    configs.append(
        FusionTestConfig(
            name="softmax.two_independent",
            func=lambda x, y: jnn.softmax(x, axis=-1) + jnn.softmax(y, axis=-1),
            args=(randn(4, 16), randn(4, 16)),
            expected_custom_calls={"mps.softmax": 2},
        )
    )
    # A plain elementwise divide — must NOT match the softmax pattern.
    configs.append(
        FusionTestConfig(
            name="softmax.not_softmax",
            func=lambda x, y: x / y,
            args=(randn(4, 8), randn(4, 8)),
            expected_custom_calls={"mps.softmax": 0},
        )
    )

    # --- fuse_layer_norm: affine LayerNorm decomposition -> mps.layer_norm ---

    import flax.linen as flax_nn

    def _flax_layer_norm(use_scale=True, use_bias=True):
        """A flax LayerNorm wrapped so its params are explicit fn args.

        Returns (func, arg_factories). The trace produces the exact
        E[x^2]-E[x]^2 + affine StableHLO the pass targets.
        """
        layer = flax_nn.LayerNorm(use_scale=use_scale, use_bias=use_bias)

        def f(x, *params):
            variables = {"params": {}}
            if use_scale:
                variables["params"]["scale"] = params[0]
            if use_bias:
                variables["params"]["bias"] = params[-1]
            return layer.apply(variables, x)

        return f

    # Full affine LayerNorm at several ranks. Default flax form (E[x^2]-E[x]^2,
    # max(0,var) clamp, use_scale=use_bias=True) — must fuse.
    for shape in [(4, 16), (2, 8, 16), (2, 3, 4, 32)]:
        d = shape[-1]
        configs.append(
            FusionTestConfig(
                name=f"layer_norm.affine.{'x'.join(map(str, shape))}",
                func=_flax_layer_norm(),
                args=(randn(*shape), randn(d), randn(d)),
                expected_custom_calls={"mps.layer_norm": 1},
                # fast::layer_norm reorders the reduction vs the decomposed
                # form; loosen the fused-vs-unfused check accordingly.
                fusion_atol=2e-4,
                fusion_rtol=2e-4,
                atol=2e-4,
                rtol=2e-4,
            )
        )

    # Tiny epsilon (1e-12). Guards the backend_config serialization: eps must be
    # written with enough precision that it doesn't round to "0.000000" (which
    # std::to_string would do). If eps were dropped to 0, the fused kernel would
    # diverge from the reference on a low-variance input. The custom flax layer
    # with epsilon set exercises this end to end.
    def _flax_layer_norm_eps(eps):
        layer = flax_nn.LayerNorm(epsilon=eps)

        def f(x, scale, bias):
            return layer.apply({"params": {"scale": scale, "bias": bias}}, x)

        return f

    configs.append(
        FusionTestConfig(
            name="layer_norm.tiny_eps",
            func=_flax_layer_norm_eps(1e-12),
            args=(randn(2, 8, 16), randn(16), randn(16)),
            expected_custom_calls={"mps.layer_norm": 1},
            fusion_atol=2e-4,
            fusion_rtol=2e-4,
            atol=2e-4,
            rtol=2e-4,
        )
    )

    # Hand-written affine LayerNorm (the mps.ops fallback form: mean of squared
    # deviation rather than E[x^2]-E[x]^2). Different variance graph — must NOT
    # fuse (conservative: we only claim the flax/E[x^2]-E[x]^2 form).
    def _manual_ln(x, w, b):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) * (var + 1e-5) ** -0.5 * w + b

    configs.append(
        FusionTestConfig(
            name="layer_norm.manual_variance_no_fuse",
            func=_manual_ln,
            args=(randn(2, 8, 16), randn(16), randn(16)),
            expected_custom_calls={"mps.layer_norm": 0},
        )
    )

    # No affine params (use_scale=use_bias=False): no weight/bias to pass to the
    # kernel — must NOT fuse.
    configs.append(
        FusionTestConfig(
            name="layer_norm.no_affine_no_fuse",
            func=_flax_layer_norm(use_scale=False, use_bias=False),
            args=(randn(2, 8, 16),),
            expected_custom_calls={"mps.layer_norm": 0},
            atol=2e-4,
            rtol=2e-4,
        )
    )

    # Plain affine over a non-trailing axis — must NOT match (kernel is last-axis).
    def _ln_axis0(x, w, b):
        mean = jnp.mean(x, axis=0, keepdims=True)
        var = jnp.mean(x * x, axis=0, keepdims=True) - mean * mean
        return (x - mean) * jax.lax.rsqrt(var + 1e-5) * w + b

    configs.append(
        FusionTestConfig(
            name="layer_norm.non_trailing_axis_no_fuse",
            func=_ln_axis0,
            args=(randn(16, 8), randn(16, 1), randn(16, 1)),
            expected_custom_calls={"mps.layer_norm": 0},
        )
    )

    # --- fuse_rms_norm: RMSNorm decomposition -> mps.rms_norm ---

    def _flax_rms_norm(use_scale=True):
        layer = flax_nn.RMSNorm(use_scale=use_scale)

        def f(x, *params):
            variables = {"params": {}}
            if use_scale:
                variables["params"]["scale"] = params[0]
            return layer.apply(variables, x)

        return f

    # Default flax RMSNorm (use_scale=True). Emits a `subtract(x, broadcast(0))`
    # centering no-op that canonicalize_broadcast_identity folds before the
    # matcher runs — must fuse.
    for shape in [(4, 16), (2, 8, 16), (2, 3, 4, 32)]:
        d = shape[-1]
        configs.append(
            FusionTestConfig(
                name=f"rms_norm.scale.{'x'.join(map(str, shape))}",
                func=_flax_rms_norm(),
                args=(randn(*shape), randn(d)),
                expected_custom_calls={"mps.rms_norm": 1},
                fusion_atol=2e-4,
                fusion_rtol=2e-4,
                atol=2e-4,
                rtol=2e-4,
            )
        )

    # No scale (use_scale=False): no weight operand for the 2-arg kernel —
    # must NOT fuse.
    configs.append(
        FusionTestConfig(
            name="rms_norm.no_scale_no_fuse",
            func=_flax_rms_norm(use_scale=False),
            args=(randn(2, 8, 16),),
            expected_custom_calls={"mps.rms_norm": 0},
            atol=2e-4,
            rtol=2e-4,
        )
    )

    # RMSNorm over a non-trailing axis — must NOT match (kernel is last-axis).
    def _rms_axis0(x, w):
        ms = jnp.mean(x * x, axis=0, keepdims=True)
        return x * jax.lax.rsqrt(ms + 1e-6) * w

    configs.append(
        FusionTestConfig(
            name="rms_norm.non_trailing_axis_no_fuse",
            func=_rms_axis0,
            args=(randn(16, 8), randn(16, 1)),
            expected_custom_calls={"mps.rms_norm": 0},
        )
    )

    # --- canonicalize_broadcast_identity: must not change numerics ---

    # x - broadcast(0) over the SAME shape folds to x (and stays numerically
    # identical). Guards the safe case.
    configs.append(
        FusionTestConfig(
            name="canon.sub_zero_same_shape",
            func=lambda x: x - jnp.zeros_like(x),
            args=(randn(4, 8),),
            expected_custom_calls={},
        )
    )

    # x - broadcast(0) where the zero is the BROADCAST-UP target (x is smaller):
    # the subtract is a broadcast of x, NOT an identity. Must stay correct
    # (numerical check vs CPU catches an unsafe fold that drops the broadcast).
    configs.append(
        FusionTestConfig(
            name="canon.sub_zero_broadcasts_x",
            func=lambda x: x - jnp.zeros((4, 8), dtype=x.dtype),
            args=(randn(8),),
            expected_custom_calls={},
        )
    )

    return configs
