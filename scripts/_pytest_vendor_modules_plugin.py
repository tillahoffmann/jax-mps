"""Pytest plugin: inject test-only ``jax._src`` modules at runtime.

A few upstream test files import helpers such as ``hypothesis_test_util`` via
``from jax._src import <name>``. Those modules live in the JAX *source* tree but
are stripped from the distributed wheel, so the import fails and the whole test
file cannot be collected.

Rather than mutate the installed package, ``scripts/run_jax_tests.py`` extracts
each missing module's source (from the cloned test repo, which is pinned to the
same tag as the installed jax) into a scratch directory and points this plugin
at it via the ``JAX_MPS_VENDORED_DIR`` environment variable. At plugin-load time
we load each ``*.py`` there as ``jax._src.<stem>`` and register it in
``sys.modules`` so the test files' imports resolve -- no changes to site-packages.

Enable with ``-p _pytest_vendor_modules_plugin``; the runner does this when
``JAX_MPS_VENDORED_DIR`` is set.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_VENDORED_DIR_ENV = "JAX_MPS_VENDORED_DIR"


def _inject_vendored_modules() -> None:
    vendored_dir = os.environ.get(_VENDORED_DIR_ENV)
    if not vendored_dir:
        return
    directory = Path(vendored_dir)
    if not directory.is_dir():
        return

    # Importing the real (installed) jax._src first ensures we extend the genuine
    # package namespace rather than accidentally shadowing it.
    import jax._src

    for src in sorted(directory.glob("*.py")):
        name = f"jax._src.{src.stem}"
        if name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(name, src)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        # Register before exec so any self-referential imports resolve.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(jax._src, src.stem, module)


# Run at import time: pytest imports plugins before collecting test modules, so
# the injected modules are in place before any ``from jax._src import ...`` runs.
_inject_vendored_modules()
