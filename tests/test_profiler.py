"""jax.profiler traces include MPS device activity (PJRT plugin profiler)."""

import glob
import os

import jax
import jax.numpy as jnp
import pytest

from .configs.util import MPS_DEVICE

pytestmark = pytest.mark.skipif(MPS_DEVICE is None, reason="requires the MPS device")


def _trace(tmp_path, fn):
    with jax.profiler.trace(str(tmp_path)):
        fn()
    (path,) = glob.glob(os.path.join(tmp_path, "**", "*.xplane.pb"), recursive=True)
    return jax.profiler.ProfileData.from_file(path)


def _mps_events(profile):
    plane = profile.find_plane_with_name("/device:GPU:0")
    assert plane is not None, [p.name for p in profile.planes]
    return {
        line.name: [(e.name, e.start_ns, e.duration_ns) for e in line.events]
        for line in plane.lines
    }


def test_trace_records_executions(tmp_path):
    def double_sum(x):
        return (x * 2).sum()

    f = jax.jit(double_sum)
    x = jnp.ones((64, 64), device=MPS_DEVICE)
    f(x).block_until_ready()  # compile outside the trace window

    profile = _trace(tmp_path, lambda: [f(x).block_until_ready() for _ in range(3)])

    lines = _mps_events(profile)
    executions = lines["MPS executions"]
    named = [e for e in executions if "double_sum" in e[0]]
    assert len(named) == 3, executions

    # Device events share the host tracer's clock: each one falls inside the
    # host-side jit call that dispatched it.
    host = profile.find_plane_with_name("/host:CPU")
    calls = [
        (e.start_ns, e.start_ns + e.duration_ns)
        for line in host.lines
        for e in line.events
        if e.name == "PjitFunction(double_sum)"
    ]
    assert calls, "no host-side PjitFunction(double_sum) events"
    for _, start_ns, duration_ns in named:
        assert duration_ns > 0
        end_ns = start_ns + duration_ns
        assert any(lo <= start_ns and end_ns <= hi for lo, hi in calls), (
            (start_ns, end_ns),
            calls,
        )
    # Each execution is broken into graph build and evaluation.
    assert len(lines["Graph build"]) == len(executions)
    assert len(lines["Evaluation"]) == len(executions)


def test_trace_window_is_bounded(tmp_path):
    f = jax.jit(lambda x: x + 1)
    x = jnp.ones(8, device=MPS_DEVICE)
    f(x).block_until_ready()  # outside any trace: must not be recorded

    profile = _trace(tmp_path / "a", lambda: f(x).block_until_ready())
    assert len(_mps_events(profile)["MPS executions"]) == 1

    # A second trace starts empty rather than accumulating the first.
    profile = _trace(
        tmp_path / "b", lambda: [f(x).block_until_ready() for _ in range(2)]
    )
    assert len(_mps_events(profile)["MPS executions"]) == 2
