"""Pytest plugin: convert genuinely-unsupportable failures into skips.

Some upstream JAX tests exercise capabilities the MPS backend fundamentally
cannot provide on a single Apple-Silicon device -- not missing handlers we
could implement, but hardware/topology limits:

  * float64        -- MLX does not support float64 on Metal.
  * sub-byte / 8-bit-float dtypes -- int4/uint4 and the float8/float4 family
                      (e4m3, e5m2, e2m1, ...) have no element type in MLX, so a
                      host array of one cannot be materialised on the device.
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
    # Catch-all for any PJRT dtype not mapped in PjrtDtypeToMlx (mlx_buffer.cc),
    # which throws this exact message from its default case. Today the only such
    # dtypes are the sub-byte ints (int4/uint4) and the float8/float4 family,
    # none of which have an MLX element type, so BufferFromHostBuffer cannot
    # materialise them. The affected tests exercise JAX's dtype system
    # (promotion, repr, views), not MPS compute correctness.
    #
    # Note this is a catch-all, not an enumerated list: if a future PJRT dtype
    # that *should* map to an existing MLX type is left out of PjrtDtypeToMlx, it
    # would also be skipped here. The skipped nodeids make such a gap visible,
    # and the fix in that case is to extend the mapping rather than rely on this.
    (
        re.compile(r"Unsupported PJRT dtype: \d+"),
        "PJRT dtype not mapped in MLX (e.g. int4/uint4/float8/float4)",
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
    #
    # replica_id has no handler: current JAX lowers axis_index to partition_id
    # (handled), leaving replica_id unreachable, so it stays skip-listed as a
    # forward-looking guard in case a future lowering revives the replica path.
    (
        re.compile(
            r"Unsupported operation\(s\): stablehlo\."
            r"(all_gather|all_to_all|collective_permute|"
            r"collective_broadcast|reduce_scatter|replica_id)"
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
