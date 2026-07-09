#!/usr/bin/env python
"""Device-free smoke test that an installed jax-mps wheel loads correctly.

The wheel is tagged ``py3-none-macosx_<ver>_arm64`` -- a single, Python-ABI
agnostic artifact meant to work under every supported CPython. That claim holds
because the plugin is a pure C PJRT dylib (only ``GetPjrtApi`` is exported, no
libpython linkage), so the same ``.dylib`` loads under any interpreter. This
script pins that: run it under each Python version against the *same* wheel and
it must pass everywhere.

It deliberately requires no Metal GPU, so it is safe to run in CI on device-less
runners. It checks packaging + dynamic loading, not compute:

  1. the shim resolves the dylib inside the installed wheel,
  2. the dylib actually ``dlopen``s (catches an arch/ABI/load-time break),
  3. its single exported entry point ``GetPjrtApi`` resolves,
  4. ``jax_plugins.mps.initialize()`` runs the exact ``xla_bridge`` registration
     path JAX uses (registration is lazy -- it does not enumerate devices).

Device enumeration and real computation need a GPU and are covered by the
MPS-gated test suite, not here.

Run against a built wheel under a specific interpreter, e.g.::

    uv run --python 3.12 --no-project --with dist/*.whl python scripts/smoke_test_wheel.py

or against the current editable install::

    uv run python scripts/smoke_test_wheel.py
"""

from __future__ import annotations

import ctypes
import sys

import jax_plugins.mps as mps


def main() -> int:
    # Failures below are deliberately left uncaught: for a CI check the full
    # traceback pinpoints what broke (a missing/mis-pointed library raises
    # MPSPluginError, ctypes.CDLL raises OSError on a load failure, resolving the
    # exported symbol raises AttributeError when it is absent, initialize raises
    # MPSPluginError). An uncaught exception still exits nonzero and fails the
    # job -- a traceback is more useful here than an opaque one-line message.

    # 1. The wheel must actually ship the dylib where the shim looks for it.
    path = mps._find_library()
    if path is None:
        print(
            "SMOKE FAIL: libpjrt_plugin_mps.dylib not found in the install",
            file=sys.stderr,
        )
        return 1

    # 2. Real ABI check: dlopen the dylib and resolve its one exported symbol
    #    (see #199 -- everything else is hidden). A Python-ABI break, arch
    #    mismatch, missing dependent library, or broken export table fails here,
    #    with no MPS device required.
    lib = ctypes.CDLL(path)
    _ = lib.GetPjrtApi  # dlsym; raises AttributeError if the entry point is absent

    # 3. Exercise the entry point JAX calls to register the plugin. This runs
    #    xla_bridge.register_plugin but does not enumerate devices, so it is
    #    GPU-free.
    mps.initialize()

    print(f"SMOKE OK: {sys.version.split()[0]} loaded {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
