"""The fused threefry2x32 lowering must actually register on MPS (issue #216).

``register_fused_ops`` routes JAX's ``threefry2x32`` PRNG primitive to a fused
Metal kernel so ``jax.random`` inside a loop body stays on the counted-loop fast
path (issue #196). Registration used to import the primitive from the private
``jax._src.prng`` module, which jax 0.10.2 removed -- inside the plugin's
``>=0.10.0,<0.11`` range. The import failure was swallowed into a ``UserWarning``
and ``jax.random`` silently fell back to the ~140-op inline expansion (~10 ms ->
~170 ms on RNG-heavy scans), with every numerical test still passing because the
fallback is correct, just slow.

These tests pin the fast path *behaviourally*: lowering a ``jax.random`` call on
MPS must emit the ``@mps.threefry2x32`` custom_call. If registration ever
no-ops again -- for any jax version in range -- the custom_call disappears and
this fails loudly instead of degrading silently.
"""

from __future__ import annotations

import jax
import pytest
from jax import random

try:
    MPS_DEVICE = jax.devices("mps")[0]
except (RuntimeError, IndexError):
    MPS_DEVICE = None

pytestmark = pytest.mark.skipif(MPS_DEVICE is None, reason="MPS device required")


def test_random_lowers_to_fused_threefry_custom_call():
    """jax.random on MPS emits @mps.threefry2x32, not the inline expansion."""

    def f(key):
        return random.normal(key, (64,))

    # Place the key on the MPS device so lowering targets MPS specifically --
    # otherwise, with multiple platforms available, jit could lower for CPU and
    # miss the custom_call even when the MPS lowering is correctly registered.
    key = jax.device_put(random.key(0), MPS_DEVICE)
    lowered = jax.jit(f).lower(key)
    text = lowered.as_text()
    assert "mps.threefry2x32" in text, (
        "jax.random did not lower to the fused @mps.threefry2x32 custom_call; "
        "the fused threefry lowering failed to register and jax.random fell back "
        "to the slow inline expansion (issue #216).\n" + text
    )


def test_threefry_primitive_registered_for_mps_platform():
    """The threefry2x32 primitive has an MPS-platform lowering registered."""
    import jax.extend.random
    from jax._src.interpreters import mlir

    threefry2x32_p = jax.extend.random.threefry2x32_p
    mps_lowerings = mlir._platform_specific_lowerings.get("mps")
    assert mps_lowerings is not None, "no MPS-platform lowerings registered at all"
    assert threefry2x32_p in mps_lowerings, (
        "threefry2x32 has no MPS-platform lowering; register_fused_ops failed to "
        "register the fused threefry path (issue #216)"
    )
