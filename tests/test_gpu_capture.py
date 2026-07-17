"""Tests for env-gated Metal GPU trace capture.

``JAX_MPS_GPU_CAPTURE=<path>`` makes the plugin wrap a bounded window of
``Execute`` dispatches in ``mlx::core::metal::start_capture``/``stop_capture``,
writing a ``.gputrace`` document openable in Xcode/Instruments.

Apple gates programmatic capture behind ``MTL_CAPTURE_ENABLED=1`` in the
environment at process start; the plugin checks for it and disables capture
with a clear message rather than crashing when it is missing. Both behaviours
are exercised here via subprocesses (the env vars must be set before the
process — and the Metal device — initialize).
"""

from __future__ import annotations

import os
import subprocess
import sys

# A minimal on-device computation: force the work onto MPS and block so the
# matmul actually dispatches (and so the captured command buffer is non-empty).
_WORKLOAD = (
    "import jax, jax.numpy as jnp;"
    "d = jax.devices('mps')[0];"
    "x = jax.device_put(jnp.ones((128, 128), jnp.float32), d);"
    "(x @ x).block_until_ready()"
)


def _run_workload(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _WORKLOAD],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_gpu_capture_writes_trace(tmp_path, mps_device):
    trace = tmp_path / "capture.gputrace"
    env = {
        **os.environ,
        "MTL_CAPTURE_ENABLED": "1",
        "JAX_MPS_GPU_CAPTURE": str(trace),
    }
    result = _run_workload(env)
    assert result.returncode == 0, result.stderr
    assert trace.exists(), f"trace not written; stderr:\n{result.stderr}"
    # A .gputrace is a bundle (directory) on macOS; require it to hold content.
    if trace.is_dir():
        assert any(trace.rglob("*")), "trace bundle is empty"
    else:
        assert trace.stat().st_size > 0, "trace file is empty"


def test_gpu_capture_without_mtl_enabled_is_graceful(tmp_path, mps_device):
    trace = tmp_path / "capture.gputrace"
    env = {**os.environ, "JAX_MPS_GPU_CAPTURE": str(trace)}
    env.pop("MTL_CAPTURE_ENABLED", None)
    result = _run_workload(env)
    # Missing MTL_CAPTURE_ENABLED must NOT fail the run: capture is disabled.
    assert result.returncode == 0, result.stderr
    assert not trace.exists(), "trace should not be written without MTL_CAPTURE_ENABLED"
    assert "MTL_CAPTURE_ENABLED" in result.stderr, (
        f"expected a message naming MTL_CAPTURE_ENABLED; stderr:\n{result.stderr}"
    )
