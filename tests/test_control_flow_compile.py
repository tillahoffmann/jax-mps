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

from .configs import OperationTestConfig

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


# ---------------------------------------------------------------------------
# Tokens (stablehlo.create_token / after_all)
# ---------------------------------------------------------------------------
# A token (!stablehlo.token) carries no data; it only threads effect ordering,
# and surfaces at the jit boundary as a returned/accepted value. The MPS backend
# represents it as a trivial scalar placeholder. (Opaque token-buffer semantics —
# e.g. rejecting numpy conversion — are a separate, deliberately out-of-scope
# follow-up.)


def test_jit_returning_token():
    """A jit that returns a token must run (create_token is handled)."""
    # Tokens are opaque (no comparable value) and get DCE'd when disconnected,
    # so they can't be exercised via an OperationTestConfig value check; mark
    # them exercised here, as test_ops.py does for composite / rng_bit_generator.
    OperationTestConfig.EXERCISED_STABLEHLO_OPS.add("stablehlo.create_token")
    OperationTestConfig.EXERCISED_STABLEHLO_OPS.add("stablehlo.after_all")
    tok = jax.jit(jax.lax.create_token)()
    # And the token round-trips as a jit argument (after_all consumes it).
    jax.block_until_ready(jax.jit(lambda t: jax.lax.after_all(t))(tok))


def test_token_threaded_with_real_output():
    """A token threaded alongside a real result must not disturb the result."""
    import jax.numpy as jnp

    def f(x):
        return x + 1.0, jax.lax.after_all(jax.lax.create_token())

    out, _tok = jax.jit(f)(jnp.float32(3.0))
    assert float(out) == 4.0


def test_grad_with_live_token():
    """Differentiating a jitted fn that keeps a token live must work.

    The token is returned as an aux output (``has_aux``) so it survives DCE and
    actually appears in the differentiated, jitted graph — exercising the token
    handlers in that context — while the gradient is taken w.r.t. the real loss.
    """
    import jax.numpy as jnp

    def f(x):
        return x * x, jax.lax.after_all(jax.lax.create_token())

    value_and_grad = jax.jit(jax.value_and_grad(f, has_aux=True))
    (value, _token), grad = value_and_grad(jnp.float32(3.0))
    assert float(value) == 9.0
    assert float(grad) == 6.0
