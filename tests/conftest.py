"""Pytest configuration and custom hooks."""

import os
import re
from pathlib import Path

import pytest

from .configs import OperationTestConfig

# Aggregated ops from all xdist workers (controller only).
_aggregated_exercised_ops: set[str] = set()


def pytest_sessionfinish(session, exitstatus):
    """On xdist workers, send exercised ops back to the controller."""
    if hasattr(session.config, "workeroutput"):
        session.config.workeroutput["exercised_ops"] = list(
            OperationTestConfig.EXERCISED_STABLEHLO_OPS
        )


def pytest_testnodedown(node, error):
    """On the controller, aggregate exercised ops from each finishing worker."""
    worker_ops = node.workeroutput.get("exercised_ops", [])
    _aggregated_exercised_ops.update(worker_ops)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Check op coverage after all tests complete (works with and without xdist)."""
    if "CI" not in os.environ:
        return

    test_mode = os.environ.get("JAX_TEST_MODE", "compare").lower()
    if test_mode == "cpu":
        return

    # Use aggregated ops from xdist workers if available, otherwise use the
    # class-level set directly (serial mode).
    if _aggregated_exercised_ops:
        exercised_ops = _aggregated_exercised_ops
    else:
        exercised_ops = OperationTestConfig.EXERCISED_STABLEHLO_OPS

    pjrt_dir = Path(__file__).parent.parent / "src/pjrt_plugin"
    assert pjrt_dir.is_dir()

    # Ops that appear in JAX's lowered IR text but never reach our dispatch loop.
    mlir_lowered_ops = {
        # CHLO ops legalized by JAX before serialization
        "chlo.acos",
        "chlo.acosh",
        "chlo.asin",
        "chlo.asinh",
        "chlo.atan",
        "chlo.atan2",
        "chlo.atanh",
        "chlo.bessel_i1e",
        "chlo.cosh",
        "chlo.digamma",
        "chlo.erf",
        "chlo.erf_inv",
        "chlo.lgamma",
        "chlo.next_after",
        "chlo.sinh",
        "chlo.square",
        "chlo.tan",
        "chlo.top_k",
        # StableHLO ops lowered to more primitive ops
        "stablehlo.broadcast",
        "stablehlo.dot",
        "stablehlo.erf",
    }

    # Discover ops from the dispatch table in mlx_executable.cc and ops/*.cc
    dispatch_pattern = re.compile(r'\{"((?:stablehlo|chlo)\.[^"]+)",\s*(?:Handle|Make)')
    op_names: set[str] = set()
    for cc_file in [pjrt_dir / "mlx_executable.cc"] + sorted(
        (pjrt_dir / "ops").glob("*.cc")
    ):
        with cc_file.open() as fp:
            op_names.update(dispatch_pattern.findall(fp.read()))

    assert op_names, "Failed to discover any ops."
    exercised = exercised_ops - mlir_lowered_ops
    unsupported = exercised - op_names
    if unsupported:
        terminalreporter.section("Op coverage errors")
        terminalreporter.write_line(
            f"ERROR: {len(unsupported)} unsupported ops: {', '.join(sorted(unsupported))}"
        )
        terminalreporter._session.exitstatus = max(
            terminalreporter._session.exitstatus, 1
        )

    missing = op_names - exercised_ops - mlir_lowered_ops
    if missing:
        terminalreporter.section("Op coverage errors")
        terminalreporter.write_line(
            f"ERROR: {len(missing)} untested ops: {', '.join(sorted(missing))}"
        )
        terminalreporter._session.exitstatus = max(
            terminalreporter._session.exitstatus, 1
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Validate xfail match patterns when tests fail as expected."""
    outcome = yield
    report = outcome.get_result()

    # Only process xfail tests that failed during the call phase
    if report.when != "call":
        return

    marker = item.get_closest_marker("xfail")
    if marker is None:
        return

    match_pattern = marker.kwargs.get("match")
    if match_pattern is None:
        return

    # Check if test raised an exception (xfail should have failed)
    if call.excinfo is None:
        return

    exc_message = str(call.excinfo.value)

    if not re.search(match_pattern, exc_message, re.MULTILINE):
        # The test failed, but not with the expected message pattern.
        # Convert this to a real failure so the user notices.
        report.outcome = "failed"
        report.longrepr = (
            f"XFAIL match failed: exception message did not match pattern.\n"
            f"  Pattern: {match_pattern!r}\n"
            f"  Message: {exc_message!r}"
        )
