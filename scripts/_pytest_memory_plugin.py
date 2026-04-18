"""Pytest plugin that records RSS memory before/after each test to a CSV.

Enable by passing ``-p _pytest_memory_plugin --memory-csv=PATH`` (with the
``scripts/`` directory on ``PYTHONPATH``; ``scripts/run_jax_tests.py`` does
this automatically when its ``--memory-csv`` flag is given).

CSV columns:
    nodeid             test node id
    rss_before_mb      RSS just before the test runs
    rss_after_mb       RSS just after the test finishes
    delta_mb           rss_after_mb - rss_before_mb
    max_after_rss_mb   running maximum of rss_after_mb across all tests so
                       far (NOT a true mid-test peak — RSS spikes that
                       drop before the test finishes are not captured)

If ``--current-test-file=PATH`` is also given, the plugin writes the nodeid of
each test *before* it runs (overwriting the file each time) and clears it after.
This makes it possible to identify which test is responsible for a hang that
prevents pytest from completing the test (and thus from writing a CSV row).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# psutil is imported lazily inside pytest_configure when --memory-csv is set,
# so --current-test-file works without it.

_MAX_AFTER_RSS_MB = 0.0


def pytest_addoption(parser):
    parser.addoption(
        "--memory-csv",
        action="store",
        default=None,
        help="Write per-test RSS memory deltas to this CSV path.",
    )
    parser.addoption(
        "--current-test-file",
        action="store",
        default=None,
        help=(
            "If set, write the nodeid of the currently running test to this "
            "file before each test starts (and clear it after each test "
            "finishes). Useful for identifying tests that hang."
        ),
    )


def pytest_configure(config):
    path = config.getoption("--memory-csv")
    if path:
        try:
            import psutil
        except ImportError as e:
            raise pytest.UsageError(
                "--memory-csv requires the 'psutil' package; install it via "
                "`uv pip install psutil` or `pip install psutil`"
            ) from e
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(
            ["nodeid", "rss_before_mb", "rss_after_mb", "delta_mb", "max_after_rss_mb"]
        )
        config._memory_fh = fh
        config._memory_writer = writer
        config._memory_proc = psutil.Process()

    cur = config.getoption("--current-test-file")
    if cur:
        cp = Path(cur)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text("")
        global _CURRENT_TEST_PATH
        _CURRENT_TEST_PATH = cp


def pytest_unconfigure(config):
    fh = getattr(config, "_memory_fh", None)
    if fh is not None:
        fh.close()


_CURRENT_TEST_PATH: Path | None = None


def pytest_runtest_logstart(nodeid, location):
    if _CURRENT_TEST_PATH is not None:
        try:
            _CURRENT_TEST_PATH.write_text(nodeid + "\n")
        except OSError:
            pass


def pytest_runtest_logfinish(nodeid, location):
    if _CURRENT_TEST_PATH is not None:
        try:
            _CURRENT_TEST_PATH.write_text("")
        except OSError:
            pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    config = item.config
    writer = getattr(config, "_memory_writer", None)
    if writer is None:
        yield
        return

    proc = config._memory_proc

    def rss_mb() -> float:
        return proc.memory_info().rss / (1024 * 1024)

    before = rss_mb()
    yield
    after = rss_mb()

    global _MAX_AFTER_RSS_MB
    if after > _MAX_AFTER_RSS_MB:
        _MAX_AFTER_RSS_MB = after

    writer.writerow(
        [
            item.nodeid,
            f"{before:.2f}",
            f"{after:.2f}",
            f"{after - before:+.2f}",
            f"{_MAX_AFTER_RSS_MB:.2f}",
        ]
    )
    config._memory_fh.flush()
