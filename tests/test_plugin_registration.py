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

import builtins

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


def test_patch_jax_functions_survives_incompatible_flax(monkeypatch):
    """A present-but-incompatible flax must be skipped, not crash patching.

    The flax LayerNorm patch is best-effort. A flax built against a different
    jax can raise at import from touching jax internals that moved -- e.g.
    ``jax.core.Effect``, removed in jax 0.11 -- which is an AttributeError, not
    an ImportError. ``patch_jax_functions`` must degrade to skipping flax rather
    than letting that escape and crash ``initialize()`` (this happened on the
    jax-nightly canary).
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flax" or name.startswith("flax."):
            raise AttributeError(
                "jax.core.Effect was deprecated in JAX v0.10.0 and removed in "
                "JAX v0.11.0."
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Must not raise: the incompatible flax import is skipped like an absent one.
    ops.patch_jax_functions()
