#!/usr/bin/env python3
"""Run upstream JAX test suite against the MPS backend and report pass rate.

Runs each test file in a separate subprocess to bound memory usage.

Usage:
    uv run python scripts/run_jax_tests.py [OPTIONS]

Options:
    --filter PATTERN    Only run test files matching glob pattern
    --results-dir DIR   Where to write results (default: /tmp/jax-test-results)
    --clone-dir DIR     Where to clone JAX tests (default: /tmp/jax-test-suite)
    --keep              Reuse existing clone
    --timeout SECONDS   Per-test timeout in seconds (default: 60)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from pathlib import Path

# Files excluded from the suite
EXCLUDED_FILES = {
    "x64_context_test.py",  # requires float64 -- MPS only supports float32
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
    if keep and tests_dir.is_dir():
        print(f"Reusing existing clone at {clone_dir}")
        return tests_dir

    tag = f"jax-v{version}"
    print(f"Cloning JAX {tag} tests to {clone_dir} ...")

    if clone_dir.exists():
        subprocess.run(["rm", "-rf", str(clone_dir)], check=True)

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


def run_file(
    test_file: Path, xml_dir: Path, timeout: int, project_root: Path
) -> dict[str, int] | None:
    """Run a single test file in a subprocess, return parsed counts or None."""
    xml_path = xml_dir / f"{test_file.stem}.xml"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        f"--junitxml={xml_path}",
        "--override-ini=addopts=",
        "-p",
        "no:faulthandler",
        "-p",
        "no:benchmark",
        "--tb=no",
        "-q",
        f"--timeout={timeout}",
        "--continue-on-collection-errors",
    ]
    env = {**os.environ, "JAX_PLATFORMS": "mps"}
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            cwd=project_root,
            timeout=60 + timeout * 100,
            env=env,
        )  # generous file-level cap
    except subprocess.TimeoutExpired:
        pass

    if xml_path.exists():
        try:
            return parse_junit_xml(xml_path)
        except ET.ParseError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run JAX upstream tests against MPS backend"
    )
    parser.add_argument("--filter", default="*_test.py", help="Glob for test files")
    parser.add_argument("--results-dir", default="/tmp/jax-test-results")
    parser.add_argument("--clone-dir", default="/tmp/jax-test-suite")
    parser.add_argument("--keep", action="store_true", help="Reuse existing clone")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-test timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    xml_dir = results_dir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = Path(args.clone_dir)

    version = get_jax_version()
    print(f"JAX version: {version}")

    tests_dir = clone_jax_tests(version, clone_dir, keep=args.keep)

    test_files = sorted(
        f
        for f in tests_dir.glob("*_test.py")
        if fnmatch(f.name, args.filter) and f.name not in EXCLUDED_FILES
    )
    if EXCLUDED_FILES:
        print(f"Excluded: {', '.join(sorted(EXCLUDED_FILES))}")
    print(f"Found {len(test_files)} test files\n")

    # Run each file in a separate subprocess
    totals = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0}
    no_result_files: list[str] = []
    for i, tf in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] {tf.name} ...", end=" ", flush=True)
        counts = run_file(tf, xml_dir, args.timeout, project_root)
        if counts is None:
            print("NO RESULTS (counted as 1 error)")
            totals["errors"] += 1
            no_result_files.append(tf.name)
            continue
        for k in totals:
            totals[k] += counts[k]
        n = sum(counts.values())
        print(
            f"P={counts['passed']} F={counts['failed']} "
            f"E={counts['errors']} S={counts['skipped']} ({n} tests)"
        )

    # Summary
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
    if no_result_files:
        print(f"\n  No results:   {', '.join(no_result_files)}")
    print("=" * 60)

    summary_path = results_dir / "summary.txt"
    summary_path.write_text(f"{passed}/{available} ({pct:.1f}%) -- JAX {version}\n")
    print(f"\nPer-file XML: {xml_dir}/")
    print(f"Summary:      {summary_path}")


if __name__ == "__main__":
    main()
