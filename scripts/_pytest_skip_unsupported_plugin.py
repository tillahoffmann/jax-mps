"""Pytest plugin: convert genuinely-unsupportable failures into skips.

Some upstream JAX tests exercise capabilities the MPS backend fundamentally
cannot provide on a single Apple-Silicon device -- not missing handlers we
could implement, but hardware/topology limits:

  * float64        -- MLX does not support float64 on Metal.
  * a CPU device   -- jax-mps exposes only the 'mps' platform, so tests that
                      request the 'cpu' backend, or that rely on a local CPU
                      device (e.g. jax.debug.callback), cannot run.
  * multiple devices -- collective ops (all_reduce, all_gather, ...) require a
                      multi-device mesh; MPS presents a single device.

These are excluded from the pass-rate denominator by turning the matching
failure into a skip, mirroring how float64 is handled. Each pattern is kept
deliberately specific so we never silently mask a *fixable* failure (a missing
op handler, a numerical bug, a Metal size limit, etc.).

Enable by passing ``-p _pytest_skip_unsupported_plugin``. The
``scripts/run_jax_tests.py`` runner does this automatically.
"""

from __future__ import annotations

import re

import pytest

# Each entry is (compiled-regex, reason). Matched against the failure
# traceback/exception text only (never captured stdout/stderr), so an
# unrelated test that merely logged one of these strings is not skipped.
_UNSUPPORTED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # MLX has no float64 on Metal. The PJRT plugin throws this exact message at
    # buffer-creation time (PjrtDtypeToMlx in mlx_buffer.cc). Kept specific so we
    # never skip a complex64 / fp16 / bf16 failure that shares MLX's generic
    # "Only float32" wording.
    (
        re.compile(r"MLX does not support float64"),
        "MLX does not support float64 on Metal",
    ),
    # Test requests a backend other than the only one jax-mps provides.
    (
        re.compile(r"Unknown backend cpu\b"),
        "jax-mps exposes only the 'mps' platform; no CPU backend available",
    ),
    # jax.debug.callback (and friends) need a local CPU device to host inputs.
    (
        re.compile(r"failed to find a local CPU device"),
        "jax-mps exposes only the 'mps' platform; no local CPU device available",
    ),
    # Collective ops require a multi-device mesh; MPS presents a single device.
    # Match the specific collective op names inside the "Unsupported operation(s)"
    # message so generic missing-handler failures stay visible.
    (
        re.compile(
            r"Unsupported operation\(s\): stablehlo\."
            r"(all_reduce|all_gather|all_to_all|collective_permute|"
            r"collective_broadcast|reduce_scatter|partition_id|replica_id)"
        ),
        "collective ops require a multi-device mesh; MPS is single-device",
    ),
]


def _match_unsupported(text: str) -> str | None:
    if not text:
        return None
    for pattern, reason in _UNSUPPORTED_PATTERNS:
        if pattern.search(text):
            return reason
    return None


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return

    # Match only against the failure traceback/exception text, NOT captured
    # stdout/stderr sections. An unrelated test that logged one of these
    # messages earlier and then failed for some other reason must NOT be
    # skipped.
    haystack = str(report.longrepr) if report.longrepr is not None else ""

    reason = _match_unsupported(haystack)
    if reason is not None:
        report.outcome = "skipped"
        # pytest renders skipped with longrepr as a (path, lineno, reason)
        # tuple; reuse the test's location.
        report.longrepr = (str(item.fspath), 0, f"Skipped: {reason}")
