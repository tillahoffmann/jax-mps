#!/usr/bin/env python
"""Device-free check that the fused-op lowerings register under the installed jax.

CI runners have no MPS GPU, so the MPS-gated test suite cannot exercise the
``platform="mps"`` lowerings. Registration itself is device-free, though:
``initialize()`` registers the ``mps`` platform via ``xla_bridge`` and then calls
``register_fused_ops``, which registers each fused lowering rule -- a dict
insertion, no device needed.

``register_fused_ops`` resolves several jax internals to reach the primitives it
lowers. When one moves between releases -- e.g. ``jax._src.prng``, removed in jax
0.10.2 (issue #216) -- the fused ``threefry2x32`` registration can silently
no-op with only a warning, dropping ``jax.random`` back to the ~140-op inline
PRNG expansion. This check turns that into a hard failure so a jax-version
matrix catches it without an MPS device.

Run against an installed wheel::

    .venv/bin/python scripts/check_jax_compat.py
"""

from __future__ import annotations

import warnings
from importlib.metadata import version

import jax
import jax.extend.random
from jax._src.interpreters import mlir

import jax_plugins.mps as mps


def main() -> int:
    print(f"jax {version('jax')}, jaxlib {version('jaxlib')}")

    # initialize() registers the "mps" platform, then runs register_fused_ops.
    # A broken jax-internal import surfaces there as a warning (not an
    # exception), so capture warnings and fail on the registration-failure one.
    # Any hard failure (module import, register_plugin) is deliberately left
    # uncaught -- the traceback pinpoints the break and still exits nonzero.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mps.initialize()

    threefry_failures = [
        str(w.message)
        for w in caught
        if "could not register fused threefry" in str(w.message)
    ]
    assert not threefry_failures, (
        "fused threefry2x32 lowering failed to register: "
        + "; ".join(threefry_failures)
    )

    # Belt and suspenders: the primitive must actually carry an mps lowering,
    # independent of whether a warning happened to be emitted.
    mps_lowerings = mlir._platform_specific_lowerings.get("mps", {})
    assert jax.extend.random.threefry2x32_p in mps_lowerings, (
        "threefry2x32 has no platform=mps lowering after initialize(); the fused "
        "path did not register (issue #216)"
    )

    print("COMPAT OK: fused threefry2x32 lowering registered for platform=mps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
