"""Pytest plugin: convert genuine F64 buffer-creation failures into skips.

MLX fundamentally does not support float64 on Metal. The PJRT plugin throws
at buffer-creation time (PjrtDtypeToMlx in mlx_buffer.cc) with the message
"MLX does not support float64 (F64)". That's the only signature that
reliably indicates a test relies on float64; anything else (e.g. MLX's
"Only float32 is supported on Metal" linalg rejection) also fires for
complex64 / float16 / bfloat16 and would mask fixable bugs.

Scope is deliberately narrow: only the F64 buffer-creation message is
converted to skipped. All other failures (complex64, int conv, missing
handlers, Metal size limits, etc.) remain failures.

Enable by passing ``-p _pytest_skip_unsupported_plugin``. The
``scripts/run_jax_tests.py`` runner does this automatically.
"""

from __future__ import annotations

import re

import pytest

# Match the exact message raised by PjrtDtypeToMlx for F64 buffers. Kept
# specific so we never silently skip a complex64 / fp16 / bf16 failure that
# happens to share MLX's generic "Only float32" wording.
_F64_PATTERNS = [
    re.compile(r"MLX does not support float64"),
]

_SKIP_REASON = "MLX does not support float64 on Metal"


def _matches_f64_rejection(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _F64_PATTERNS)


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return

    haystack_parts = []
    if report.longrepr is not None:
        haystack_parts.append(str(report.longrepr))
    for _, content in report.sections:
        if content:
            haystack_parts.append(content)
    haystack = "\n".join(haystack_parts)

    if _matches_f64_rejection(haystack):
        report.outcome = "skipped"
        # pytest renders skipped with longrepr as a (path, lineno, reason)
        # tuple; reuse the test's location.
        report.longrepr = (str(item.fspath), 0, f"Skipped: {_SKIP_REASON}")
