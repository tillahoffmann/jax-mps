"""Regression test: nested ``stablehlo.case`` must take the MLX compile path.

A ``case`` (from ``lax.cond`` / ``lax.switch``) whose selector is itself produced
by another ``case`` — a pattern NumPyro's NUTS sampler generates — used to bail
the MLX compile path in ``HandleCase`` with "unrecognized index pattern" and
silently fall back to the slower eager execution path. The selector is passed to
``CasePrimitive`` as a ``make_arrays()`` graph edge, so it is recomputed on every
replay (never frozen) — the same property that makes ``WhileLoopPrimitive``'s
external captures safe — so the bail was unnecessary.

This is not a correctness failure (the eager fallback produces correct results);
the regression is that the compile path is abandoned. We assert the bail message
does not appear, and that inference still produces finite, sane results without
deadlocking (a real risk for these re-entrant control-flow primitives).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

try:
    MPS_DEVICE = jax.devices("mps")[0]
except (RuntimeError, IndexError):
    MPS_DEVICE = None

# NumPyro is the reliable source of the nested-case pattern (it survives JAX's
# own folding and only loses its index clamp in our simplification pass).
numpyro = pytest.importorskip("numpyro")

pytestmark = pytest.mark.skipif(MPS_DEVICE is None, reason="MPS device required")

# Substring of the HandleCase compile-path bail (control_flow.cc). Its presence
# means a case fell back to eager instead of compiling.
_BAIL_MESSAGE = "unrecognized index pattern"


def _run_gp_nuts():
    """Tiny GP regression with NUTS; the kernel cholesky + NUTS integrator
    produce a ``case``-selects-``case`` graph. Returns posterior samples."""
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    rng = np.random.default_rng(0)
    n = 12
    x = np.sort(rng.random(n)).astype(np.float32)
    y = (np.sin(3 * x) + 0.1 * rng.standard_normal(n)).astype(np.float32)

    def model(x, y=None):
        var = numpyro.sample("var", dist.LogNormal(0.0, 1.0))
        ls = numpyro.sample("ls", dist.LogNormal(0.0, 1.0))
        noise = numpyro.sample("noise", dist.HalfNormal(0.5))
        d = (x[:, None] - x[None, :]) ** 2
        k = var * jnp.exp(-0.5 * d / ls**2) + (noise + 1e-5) * jnp.eye(n)
        numpyro.sample(
            "y",
            dist.MultivariateNormal(
                loc=jnp.zeros(n), scale_tril=jnp.linalg.cholesky(k)
            ),
            obs=y,
        )

    mcmc = MCMC(
        NUTS(model), num_warmup=5, num_samples=5, num_chains=1, progress_bar=False
    )
    mcmc.run(jax.random.key(0), x, y)
    return mcmc.get_samples()


@pytest.mark.timeout(180)
def test_nested_case_compiles(capfd):
    """A nested ``stablehlo.case`` must compile rather than bail to eager."""
    samples = _run_gp_nuts()
    captured = capfd.readouterr()

    # Inference must produce finite posterior samples (correctness + no deadlock;
    # a deadlock would trip the timeout above).
    for name, value in samples.items():
        assert np.all(np.isfinite(np.asarray(value))), f"non-finite samples: {name}"

    # The compile-path bail must not fire: the nested case should compile.
    assert _BAIL_MESSAGE not in captured.err, (
        "nested stablehlo.case fell back to eager execution (compile bail):\n"
        f"{captured.err}"
    )
