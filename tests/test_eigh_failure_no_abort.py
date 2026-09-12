"""A failing MLX op must raise a catchable error, not abort the process.

MLX evaluates an op's ``eval_cpu``/``eval_gpu`` work in a lambda dispatched onto
a scheduler *stream worker thread*. Stock MLX runs that lambda with no
exception handling (``StreamThread::thread_fn`` calls ``task()`` bare), so when
an op throws on the worker thread the exception unwinds off the top of the
thread -> ``std::terminate`` -> ``abort()``, killing the whole process. Our
``Execute`` try/catch only guards the dispatch thread and cannot catch this.

The trigger here is ``eigh`` on a non-finite input: LAPACK ``syevd`` returns
``info != 0`` and ``eigh_impl`` throws ``std::runtime_error`` inside its
dispatched lambda.

This used to drive the same failure through LOBPCG on an ill-conditioned
(cond 1e5) operator, mirroring the upstream JAX suite's
``lobpcg_test.py::...geom_cond_100k`` abort. That trigger turned out to be a
symptom of #232 rather than genuine non-convergence: LOBPCG's Rayleigh-Ritz
step reverses its basis (``jax/experimental/sparse/linalg.py``: ``V[:, ::-1]``),
and a reversed view feeding a fused kernel read as zeros past the first
element, so the *next* iteration got a degenerate projection. MLX v0.32.0
fixed that, LOBPCG converges, and the test silently stopped exercising the
worker-thread path. Feed ``eigh`` a matrix LAPACK must reject instead, so the
trigger does not depend on a miscompile.

A vendored MLX patch (``third_party/mlx/patches/10-...``) catches the worker
exception and re-throws it at the next synchronization point, turning the abort
into an ordinary Python exception. This test pins that behaviour. It runs in a
subprocess because, pre-fix, the failure is a process-level ``abort()`` that no
in-process ``try/except`` can catch.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Drive MLX eigh to a failure LAPACK cannot avoid: an all-NaN symmetric matrix
# makes syevd return info != 0, and `Eigh::eval_cpu` throws inside its
# dispatched lambda. The computation is wrapped in try/except: post-fix the
# failure is a catchable Python exception, so the process exits 0 having
# printed NOABORT:RAISED with the MLX error message; pre-fix the process
# aborts (SIGABRT) before any print. A NaN on the diagonal alone is not
# enough -- syevd converges and returns NaN eigenvalues.
_WORKLOAD = (
    "import os; os.environ.setdefault('JAX_PLATFORMS', 'mps');"
    "import numpy as np, jax, jax.numpy as jnp;"
    "a = jnp.asarray(np.full((8, 8), np.nan, dtype=np.float32));"
    "\ntry:\n"
    "    w, v = jnp.linalg.eigh(a);\n"
    "    jax.block_until_ready((w, v));\n"
    "    print('NOABORT:COMPLETED')\n"
    "except Exception as e:\n"
    "    print('NOABORT:RAISED', repr(str(e)))\n"
)


def test_failing_eigh_raises_instead_of_aborting(mps_device):
    result = subprocess.run(
        [sys.executable, "-c", _WORKLOAD],
        env={**os.environ, "JAX_PLATFORMS": "mps"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    # The op failure must surface as a normal Python exception, so the process
    # exits cleanly (returncode 0) rather than aborting. SIGABRT shows up as
    # returncode -6.
    assert result.returncode == 0, (
        f"process did not exit cleanly (returncode={result.returncode}); a "
        f"failing MLX op should raise, not abort.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # Require the raised path specifically: a silent NOABORT:COMPLETED would
    # mean the failing eigh path stopped triggering and the test no longer
    # exercises the regression.
    assert "NOABORT:RAISED" in result.stdout, (
        f"expected the failure to surface as a catchable exception, but it did "
        f"not (no NOABORT:RAISED).\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # And that it is the worker-thread eigh failure that surfaced, not some
    # unrelated error.
    assert "Eigenvalue decomposition failed" in result.stdout, (
        f"raised exception did not carry the MLX eigh error message.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "terminating due to uncaught exception" not in result.stderr, (
        f"MLX still terminated on an uncaught worker-thread exception.\n"
        f"stderr:\n{result.stderr}"
    )
