#!/usr/bin/env python3
"""Run upstream JAX test suite against the MPS backend and report pass rate.

Usage:
    uv run python scripts/run_jax_tests.py [OPTIONS]

Options:
    --results-dir DIR   Where to write results (default: /tmp/jax-test-results)
    --clone-dir DIR     Where to unpack JAX tests (default: /tmp/jax-test-suite)
    --keep              Reuse existing download
    --timeout SECONDS   Per-test timeout in seconds (default: 60)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Upstream JAX repo; tests are fetched as a hash/tag-pinned source tarball
# (no git clone -- see fetch_jax_tests).
JAX_REPO = "jax-ml/jax"

# Files / directories excluded from the suite
EXCLUDED_FILES = {
    "x64_context_test.py",  # requires float64 -- MPS only supports float32
    "pallas",  # Pallas kernel-authoring path is not implemented for MPS;
    #          interpret-mode tests hang the runner without firing the
    #          per-test --timeout (native-code stall).
    "mosaic",  # Mosaic GPU/TPU kernel-authoring tests target CUDA/TPU; the
    #          MPS backend cannot run them (and several import test-only
    #          modules absent from the jax wheel).
    "multiprocess",  # Multi-host tests require multiple processes/devices;
    #          MPS presents a single device on one host.
    "documentation_coverage_test.py",  # Reads JAX's docs/ RST tree (e.g.
    #          docs/jax.numpy.rst), which the tests-only runner does not fetch;
    #          they validate JAX's documentation coverage, not the MPS backend.
}

# Test-only modules that live in the JAX source tree but are NOT shipped in the
# jax wheel. Tests import them via ``from jax._src import <name>``; without them
# the whole module fails to collect. We extract each from the source tarball
# (pinned to the same tag as the installed jax) and inject it into sys.modules at
# runtime via _pytest_vendor_modules_plugin -- site-packages is left untouched.
VENDORED_TEST_MODULES = ("hypothesis_test_util",)


def get_jax_version() -> str:
    result = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _jax_tarball(version: str, clone_dir: Path) -> Path:
    """Download (and cache) the hash-pinned JAX source tarball for jax-v<version>.

    GitHub serves a deterministic archive of any tag's tree, so there is no git
    clone -- and thus no .git directory for a reaped /tmp to leave half-populated
    (the failure mode a shallow clone hits when its objects are GC'd). Cached
    under ``clone_dir/.cache`` and keyed by tag.
    """
    tag = f"jax-v{version}"
    cache = clone_dir / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / f"{tag}.tar.gz"
    if not tarball.is_file():
        url = f"https://github.com/{JAX_REPO}/archive/refs/tags/{tag}.tar.gz"
        print(f"Downloading {url} ...")
        # Download to a temp name then rename so an interrupted download never
        # leaves a truncated tarball that looks complete on the next run.
        tmp = tarball.with_suffix(".tmp")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:  # noqa: S310
            shutil.copyfileobj(resp, f)
        tmp.rename(tarball)
    return tarball


def _iter_subtree(tf: tarfile.TarFile, prefix: str):
    """Yield members under ``<top>/<prefix>``, renamed relative to <top>.

    GitHub archives nest everything under a single top-level ``<repo>-<ref>/``
    directory; we strip it. ``prefix`` is matched on the post-strip path so only
    the top-level ``tests/`` is selected, not nested ones (e.g. examples/*/tests).
    """
    for member in tf.getmembers():
        parts = member.name.split("/", 1)
        if len(parts) < 2:
            continue
        rel = parts[1]
        if rel == prefix or rel.startswith(prefix + "/"):
            member.name = rel
            yield member


def fetch_jax_tests(version: str, clone_dir: Path, keep: bool) -> Path:
    """Fetch the upstream JAX test tree from the source tarball (no git)."""
    tests_dir = clone_dir / "tests"
    tag = f"jax-v{version}"
    stamp = clone_dir / ".jax-tag"

    if (
        keep
        and tests_dir.is_dir()
        and stamp.is_file()
        and stamp.read_text().strip() == tag
    ):
        print(f"Reusing existing JAX tests at {clone_dir} ({tag})")
        return tests_dir

    print(f"Fetching JAX {tag} tests to {clone_dir} ...")
    # Replace only the extracted tree; keep the tarball cache under .cache.
    if tests_dir.exists():
        shutil.rmtree(tests_dir)
    stamp.unlink(missing_ok=True)
    clone_dir.mkdir(parents=True, exist_ok=True)

    tarball = _jax_tarball(version, clone_dir)
    with tarfile.open(tarball) as tf:
        # Extract only the top-level tests/ subtree. filter="data" guards against
        # path-traversal/special members (Python 3.12+ default, set explicitly).
        tf.extractall(clone_dir, members=_iter_subtree(tf, "tests"), filter="data")

    # Fail hard if the archive yielded no tests: a structure change or a bad/empty
    # download must not silently degrade into a "0 collected, 100% pass" run.
    n_tests = sum(1 for _ in tests_dir.glob("*_test.py"))
    if n_tests == 0:
        raise RuntimeError(
            f"No '*_test.py' files extracted from {tarball} into {tests_dir}; "
            "the JAX source archive layout may have changed. Refusing to proceed."
        )

    stamp.write_text(tag + "\n")
    print(f"Fetched {n_tests} test files.")
    return tests_dir


def extract_vendored_modules(
    version: str, clone_dir: Path, dest_dir: Path
) -> Path | None:
    """Extract test-only ``jax._src`` modules omitted from the wheel.

    Modules like ``hypothesis_test_util`` exist in the JAX source tree but are
    stripped from the distributed wheel, so ``from jax._src import ...`` fails and
    the importing test files cannot be collected. We pull each from the same
    hash-pinned source tarball (matching the installed jax's tag) into
    ``dest_dir``, from where ``_pytest_vendor_modules_plugin`` injects it into
    ``sys.modules`` at runtime. The installed site-packages is never modified.

    Returns the directory containing the extracted modules, or None if nothing
    could be extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"jax/_src/{name}.py": name for name in VENDORED_TEST_MODULES}
    extracted = 0
    with tarfile.open(_jax_tarball(version, clone_dir)) as tf:
        for member in tf.getmembers():
            # Strip the top-level <repo>-<ref>/ component and match by suffix.
            rel = member.name.split("/", 1)[-1]
            name = wanted.get(rel)
            if name is None:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with src:
                (dest_dir / f"{name}.py").write_bytes(src.read())
            extracted += 1
    if extracted:
        print(f"  extracted {extracted} vendored test module(s) to {dest_dir}")
        return dest_dir
    print("  warning: no vendored test modules extracted (tests may not collect)")
    return None


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

    tests_dir = fetch_jax_tests(version, clone_dir, keep=args.keep)

    # Extract test-only jax._src modules (absent from the wheel); the vendor
    # plugin injects them into sys.modules at runtime so the importing test
    # files can be collected, without touching site-packages.
    vendored_dir = extract_vendored_modules(
        version, clone_dir, results_dir / "vendored"
    )

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
        *(["-p", "_pytest_vendor_modules_plugin"] if vendored_dir else []),
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
    # Point the vendor plugin at the extracted test-only modules (if any).
    if vendored_dir:
        env["JAX_MPS_VENDORED_DIR"] = str(vendored_dir)
    # Make scripts/ plugins (_pytest_skip_unsupported_plugin,
    # _pytest_vendor_modules_plugin, and the optional _pytest_memory_plugin)
    # importable.
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
