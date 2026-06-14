#!/usr/bin/env python3
"""Run upstream JAX test suite against the MPS backend and report pass rate.

Usage:
    uv run python scripts/run_jax_tests.py [OPTIONS]

Options:
    --results-dir DIR   Where to write results (default: /tmp/jax-test-results)
    --clone-dir DIR     Where to clone JAX tests (default: /tmp/jax-test-suite)
    --keep              Reuse existing clone
    --timeout SECONDS   Per-test timeout in seconds (default: 60)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Files / directories excluded from the suite
EXCLUDED_FILES = {
    "x64_context_test.py",  # requires float64 -- MPS only supports float32
    "pallas",  # Pallas kernel-authoring path is not implemented for MPS;
    #          interpret-mode tests hang the runner without firing the
    #          per-test --timeout (native-code stall).
}


def get_jax_version() -> str:
    result = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def clone_jax_tests(version: str, clone_dir: Path, keep: bool) -> Path:
    tests_dir = clone_dir / "tests"
    tag = f"jax-v{version}"

    if keep and tests_dir.is_dir():
        local = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        remote = (
            subprocess.run(
                ["git", "ls-remote", "origin", f"refs/tags/{tag}"],
                cwd=clone_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.split("\t", 1)[0]
            .strip()
        )
        if local and remote and local == remote:
            print(f"Reusing existing clone at {clone_dir} (HEAD={local[:7]} == {tag})")
            return tests_dir
        print(
            f"Existing clone at {clone_dir} is out of date "
            f"(HEAD={local[:7] or '?'}, remote {tag}={remote[:7] or '?'}); re-cloning."
        )

    print(f"Cloning JAX {tag} tests to {clone_dir} ...")

    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            "--no-checkout",
            "https://github.com/jax-ml/jax.git",
            str(clone_dir),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "tests"],
        cwd=clone_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout"],
        cwd=clone_dir,
        check=True,
        capture_output=True,
    )

    print(f"Cloned {sum(1 for _ in tests_dir.glob('*_test.py'))} test files.")
    return tests_dir


def parse_junit_xml(xml_path: Path) -> dict[str, int]:
    """Parse a JUnit XML file and return test outcome counts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    passed = failed = errors = skipped = xfailed = 0
    for tc in root.iter("testcase"):
        children = {child.tag for child in tc}
        if "failure" in children:
            failed += 1
        elif "error" in children:
            errors += 1
        elif "skipped" in children:
            skip_el = tc.find("skipped")
            if skip_el is not None and skip_el.get("type") == "pytest.xfail":
                xfailed += 1
            else:
                skipped += 1
        else:
            passed += 1

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "xfailed": xfailed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run JAX upstream tests against MPS backend"
    )
    parser.add_argument("--results-dir", default="/tmp/jax-test-results")
    parser.add_argument("--clone-dir", default="/tmp/jax-test-suite")
    parser.add_argument("--keep", action="store_true", help="Reuse existing clone")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-test timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--memory-csv",
        default=None,
        help="If set, record per-test RSS memory to this CSV path.",
    )
    parser.add_argument(
        "--current-test-file",
        default=None,
        help=(
            "If set, write the nodeid of the currently running test to this "
            "file before each test starts. Useful for diagnosing hangs."
        ),
    )
    args = parser.parse_args()

    # Line-buffer our own output so it stays in chronological order relative to
    # the pytest child's stream. Without this, parent print()s are block-
    # buffered when redirected to a file and only flush at exit, landing AFTER
    # all of pytest's output — so a native crash (e.g. an MLX op throwing on a
    # stream thread -> abort) leaves a log whose tail does not reflect what was
    # actually running.
    for stream in (sys.stdout, sys.stderr):
        # reconfigure() exists on TextIOWrapper (the usual stdout/stderr) but
        # not on every TextIO; getattr keeps the type checker happy and is a
        # no-op if the stream was replaced with something that lacks it.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)

    project_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    xml_dir = results_dir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = Path(args.clone_dir)

    # Always track the currently-running test. If pytest dies without writing
    # JUnit XML (native crash/abort), this file still holds the nodeid of the
    # test that was running, so we can name the culprit immediately instead of
    # bisecting. Defaults under results_dir; overridable via --current-test-file.
    current_test_file = args.current_test_file or str(results_dir / "current_test.txt")

    version = get_jax_version()
    print(f"JAX version: {version}")

    tests_dir = clone_jax_tests(version, clone_dir, keep=args.keep)

    # Build ignore list for excluded files
    ignore_args: list[str] = []
    for name in sorted(EXCLUDED_FILES):
        ignore_args.extend(["--ignore", str(tests_dir / name)])
    if EXCLUDED_FILES:
        print(f"Excluded: {', '.join(sorted(EXCLUDED_FILES))}")

    xml_path = xml_dir / "results.xml"

    # Remove stale results from a previous run
    xml_path.unlink(missing_ok=True)

    # Run pytest directly on the test suite
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tests_dir),
        *ignore_args,
        f"--junitxml={xml_path}",
        "--override-ini=addopts=",
        "-p",
        "no:faulthandler",
        "-p",
        "no:benchmark",
        "-p",
        "_pytest_skip_unsupported_plugin",
        "--tb=no",
        "-q",
        f"--timeout={args.timeout}",
        "--continue-on-collection-errors",
        # Track the in-progress test (see current_test_file above) so a crash
        # that prevents JUnit XML from being written can still be attributed.
        "-p",
        "_pytest_memory_plugin",
        f"--current-test-file={current_test_file}",
    ]
    if args.memory_csv:
        cmd += [f"--memory-csv={args.memory_csv}"]
    # Cap MLX's buffer cache for the test-suite scenario: thousands of
    # unrelated computations in one process otherwise leave freed MTLBuffers
    # resident until they swap (see #134, #139). Real workloads keep MLX's
    # tuned default; only the test runner sets this.
    env = {
        **os.environ,
        # Unbuffered child stdout/stderr so pytest's progress (and any crash
        # output) reaches the log live rather than in deferred blocks.
        "PYTHONUNBUFFERED": "1",
        "JAX_PLATFORMS": "mps",
        "JAX_MPS_CACHE_LIMIT_BYTES": os.environ.get(
            "JAX_MPS_CACHE_LIMIT_BYTES", str(1 << 30)
        ),
    }
    # Make scripts/ plugins (_pytest_skip_unsupported_plugin and the optional
    # _pytest_memory_plugin) importable.
    scripts_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = (scripts_dir + os.pathsep + env.get("PYTHONPATH", "")).rstrip(
        os.pathsep
    )
    print(f"\nRunning: pytest {tests_dir} ...")
    print(f"Current-test tracker: {current_test_file}\n")
    subprocess.run(cmd, cwd=project_root, env=env)

    # Parse results
    if not xml_path.exists():
        print("\nERROR: No JUnit XML produced — pytest may have crashed.")
        try:
            last = Path(current_test_file).read_text().strip()
        except OSError:
            last = ""
        if last:
            print(f"Last test running when pytest died: {last}")
        else:
            print(
                "No in-progress test was recorded — the crash happened during "
                "collection/teardown or before the first test started "
                f"(tracker: {current_test_file})."
            )
        sys.exit(1)

    totals = parse_junit_xml(xml_path)
    total = sum(totals.values())
    available = total - totals["skipped"] - totals["xfailed"]
    passed = totals["passed"]
    pct = (passed / available * 100) if available else 0

    print()
    print("=" * 60)
    print(f"JAX {version} Test Suite -- MPS Backend Results")
    print("=" * 60)
    print(f"  Collected:    {total}")
    print(f"  Skipped:      {totals['skipped']}  (excluded)")
    print(f"  XFailed:      {totals['xfailed']}  (excluded)")
    print(f"  Available:    {available}  (collected - skipped - xfailed)")
    print()
    print(f"  Passed:       {totals['passed']}")
    print(f"  Failed:       {totals['failed']}")
    print(f"  Errors:       {totals['errors']}")
    print()
    print(f"  Pass rate:    {passed}/{available} = {pct:.1f}%")
    print("=" * 60)

    summary_path = results_dir / "summary.txt"
    summary_path.write_text(f"{passed}/{available} ({pct:.1f}%) -- JAX {version}\n")
    print(f"\nJUnit XML: {xml_path}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
