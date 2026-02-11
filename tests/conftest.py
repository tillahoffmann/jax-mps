"""Pytest configuration and custom hooks."""

import re

import pytest


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
