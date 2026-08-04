"""Tests for env-gated Metal GPU trace capture.

``JAX_MPS_GPU_CAPTURE=<path>`` makes the plugin wrap a bounded window of
``Execute`` dispatches in ``mlx::core::metal::start_capture``/``stop_capture``,
writing a ``.gputrace`` document openable in Xcode/Instruments.

Apple gates programmatic capture behind ``MTL_CAPTURE_ENABLED=1`` in the
environment at process start; the plugin checks for it and disables capture
with a clear message rather than crashing when it is missing. Both behaviours
are exercised here via subprocesses (the env vars must be set before the
process — and the Metal device — initialize).

The capture-enabled case only runs on GitHub Actions or under
``JAX_MPS_TEST_GPU_CAPTURE=1``; see ``_RUN_CAPTURE`` below.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# Programmatic capture makes Apple's GPU tools call task_for_pid() on the
# workload, which needs the system.privilege.taskport.debug right. On a local
# machine that pops a blocking "Developer Tools Access" authorization dialog on
# every run; the GitHub Actions runners are already authorized, so keep the
# coverage there and opt in explicitly when running it by hand.
#
# Key this on GITHUB_ACTIONS rather than CI: the pre-commit pytest hook runs as
# `CI=true uv run pytest`, so a CI check would still prompt on every commit.
_RUN_CAPTURE = bool(os.environ.get("GITHUB_ACTIONS")) or bool(
    os.environ.get("JAX_MPS_TEST_GPU_CAPTURE")
)

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


@pytest.mark.skipif(
    not _RUN_CAPTURE,
    reason=(
        "needs system.privilege.taskport.debug; prompts for Developer Tools "
        "Access locally. Set JAX_MPS_TEST_GPU_CAPTURE=1 to run it."
    ),
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
