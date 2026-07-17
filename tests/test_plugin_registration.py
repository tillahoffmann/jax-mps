"""The plugin must initialize without silently degrading.

``register_fused_ops`` falls back to slow-but-correct behavior -- and only
*warns* -- when a JAX internal it depends on moves between releases (e.g.
``jax._src.prng``, removed in jax 0.10.2, issue #216). ``pytest.ini`` promotes
any ``jax-mps`` warning to an error, so this test -- which drives the plugin's
full ``initialize()`` registration path -- fails loudly on any such fallback
instead of shipping it.

This is intentionally generic: it asserts nothing about a specific op, so it
covers the next silent-registration break for free rather than growing a new
per-incident check. Registration is device-free (``initialize()`` registers the
``mps`` platform and the fused lowerings without enumerating a GPU), so it runs
on CPU-only CI where the MPS-gated suite is skipped.
"""

from __future__ import annotations

import jax_plugins.mps
from jax_plugins.mps import ops


def test_plugin_initializes_without_degradation():
    # A jax-mps degradation warning here is promoted to an error by
    # pytest.ini's filterwarnings, failing the test.
    jax_plugins.mps.initialize()

    # Sanity that initialize() actually registered the platform rather than
    # no-opping -- so a future change that stops registering entirely (without
    # warning) still fails here. Reuse the mlir handle already imported (and
    # justified) in ops rather than adding another jax._src import surface.
    assert "mps" in ops.mlir._platform_specific_lowerings
