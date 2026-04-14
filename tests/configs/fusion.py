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

    return configs
