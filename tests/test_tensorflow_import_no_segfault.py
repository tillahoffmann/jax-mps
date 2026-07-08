"""Importing TensorFlow before the first MPS compile must not segfault.

The plugin statically links MLIR/StableHLO/LLVM. If those symbols are exported
from ``libpjrt_plugin_mps.dylib`` with default (global) visibility, they can
interpose with the copies of the same symbols inside TensorFlow's MLIR
libraries once TensorFlow is imported into the same process. When the plugin
then registers its dialects on the first JAX MPS compile, the mixed symbol set
crashes inside ``mlir::MLIRContext::getOrLoadDialect`` -> SIGSEGV (see #198).

The fix marks every symbol in the dylib hidden and re-exports only the PJRT
entry point ``GetPjrtApi``, so the plugin's MLIR/LLVM symbols are never
globally visible and cannot interpose. This test pins that behaviour.

It runs in a subprocess because, pre-fix, the failure is a process-level
segfault (SIGSEGV, returncode -11) that no in-process ``try/except`` can catch.
TensorFlow must be imported *before* the first MPS compile to trigger it.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import jax
import pytest

try:
    MPS_DEVICE = jax.devices("mps")[0]
except (RuntimeError, IndexError):
    MPS_DEVICE = None

_HAVE_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None

pytestmark = [
    pytest.mark.skipif(MPS_DEVICE is None, reason="MPS device required"),
    pytest.mark.skipif(not _HAVE_TENSORFLOW, reason="tensorflow not installed"),
]

# Import TensorFlow first (pulling its MLIR libraries into the process), then
# force a JAX MPS compile via PRNGKey (threefry). Post-fix the plugin's MLIR
# symbols are hidden, so dialect loading is isolated and the process prints
# NOSEGFAULT:OK and exits 0; pre-fix it segfaults inside dialect loading before
# any print.
_WORKLOAD = (
    "import tensorflow as tf;"  # noqa: F401 -- imported for its side effects
    "import jax;"
    "key = jax.random.PRNGKey(0);"
    "jax.block_until_ready(key);"
    "print('NOSEGFAULT:OK', key)"
)


def test_tensorflow_import_before_compile_does_not_segfault():
    result = subprocess.run(
        [sys.executable, "-c", _WORKLOAD],
        env={**os.environ, "JAX_PLATFORMS": "mps"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    # A segfault surfaces as returncode -11 (SIGSEGV); any nonzero code means
    # the process did not survive importing TensorFlow before the first compile.
    assert result.returncode == 0, (
        f"process did not exit cleanly (returncode={result.returncode}); "
        f"importing TensorFlow before the first MPS compile must not crash.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Require the marker specifically, so a future change that makes the compile
    # silently no-op does not let this test pass without exercising the path.
    assert "NOSEGFAULT:OK" in result.stdout, (
        f"expected the MPS compile to complete after importing TensorFlow, but "
        f"the marker was not printed.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
