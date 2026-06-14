"""A failing MLX op must raise a catchable error, not abort the process.

MLX evaluates an op's ``eval_cpu``/``eval_gpu`` work in a lambda dispatched onto
a scheduler *stream worker thread*. Stock MLX runs that lambda with no
exception handling (``StreamThread::thread_fn`` calls ``task()`` bare), so when
an op throws on the worker thread the exception unwinds off the top of the
thread -> ``std::terminate`` -> ``abort()``, killing the whole process. Our
``Execute`` try/catch only guards the dispatch thread and cannot catch this.

The trigger here is ``eigh`` on a non-convergent input: LAPACK ``syevd`` returns
``info != 0`` and ``eigh_impl`` throws ``std::runtime_error`` inside its
dispatched lambda. LOBPCG reaches it via its Rayleigh-Ritz step on an
ill-conditioned (cond 1e5) projection -- this is the exact computation that
aborted the upstream JAX suite at ``lobpcg_test.py::...geom_cond_100k``.

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

import jax
import pytest

try:
    MPS_DEVICE = jax.devices("mps")[0]
except (RuntimeError, IndexError):
    MPS_DEVICE = None

pytestmark = pytest.mark.skipif(MPS_DEVICE is None, reason="MPS device required")

# Drive MLX eigh to a non-convergent input via LOBPCG on a diagonal operator
# with a geometric spectrum spanning 1..1e5 (condition number 1e5). The
# Rayleigh-Ritz eigh on the float32 projection fails (syevd info != 0). The
# computation is wrapped in try/except: post-fix the failure is a catchable
# Python exception, so the process exits 0 having printed NOABORT:RAISED with
# the MLX error message; pre-fix the process aborts (SIGABRT) before any print.
_WORKLOAD = (
    "import os; os.environ.setdefault('JAX_PLATFORMS', 'mps');"
    "import numpy as np, jax, jax.numpy as jnp;"
    "from jax.experimental.sparse import linalg as splinalg;"
    "n = 100;"
    "diagonal = np.logspace(0, 5, n).astype(np.float32);"
    "X = jax.random.normal(jax.random.PRNGKey(0), (n, 10), dtype=jnp.float32);"
    "f = lambda Z: diagonal[:, None] * Z;"
    "\ntry:\n"
    "    theta, _, _ = splinalg.lobpcg_standard(f, X, m=20);\n"
    "    jax.block_until_ready(theta);\n"
    "    print('NOABORT:COMPLETED')\n"
    "except Exception as e:\n"
    "    print('NOABORT:RAISED', repr(str(e)))\n"
)


def test_failing_eigh_raises_instead_of_aborting():
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
