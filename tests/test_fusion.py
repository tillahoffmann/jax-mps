"""Fusion-pass tests.

Each config declares a JAX function, args, and an expected set of `@mps.*`
custom_calls the plugin's fusion passes should produce. The harness:

1. Runs the function on MPS with JAX_MPS_DUMP_OPTIMIZED_IR set to a tmp dir,
   so the plugin writes its post-pass module IR to a file.
2. Reads the dumped files, counts `stablehlo.custom_call @mps.<name>`
   occurrences, and asserts they match the config's `expected_custom_calls`.
3. Computes the CPU reference and asserts numerical allclose.

Add new fusions by appending to `make_fusion_configs()` in
`tests/configs/fusion.py`. No new boilerplate per case.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import numpy
import pytest

from .configs import FusionTestConfig, make_fusion_configs

CUSTOM_CALL_RE = re.compile(r"stablehlo\.custom_call\s+@(mps\.[\w\.]+)")

try:
    MPS_DEVICE = jax.devices("mps")[0]
except (RuntimeError, IndexError):
    MPS_DEVICE = None

pytestmark = pytest.mark.skipif(MPS_DEVICE is None, reason="MPS device required")


@pytest.fixture(params=make_fusion_configs(), ids=lambda c: c.name)
def config(request: pytest.FixtureRequest) -> FusionTestConfig:
    return request.param


def _count_custom_calls(dump_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in sorted(dump_dir.glob("module_*.mlir")):
        for name in CUSTOM_CALL_RE.findall(p.read_text()):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _run_mps(
    config: FusionTestConfig,
    *,
    dump_dir: Path | None,
    disable_fusions: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> jax.Array:
    if dump_dir is not None:
        monkeypatch.setenv("JAX_MPS_DUMP_OPTIMIZED_IR", str(dump_dir))
    else:
        monkeypatch.delenv("JAX_MPS_DUMP_OPTIMIZED_IR", raising=False)
    if disable_fusions:
        monkeypatch.setenv("JAX_MPS_NO_OPTIMIZE", "1")
    else:
        monkeypatch.delenv("JAX_MPS_NO_OPTIMIZE", raising=False)
    # JAX caches compiled executables by (function, device, abstract args), so
    # back-to-back calls with the same function would reuse the fused artifact
    # regardless of our env vars. Clear the cache so the plugin actually
    # re-parses under the current JAX_MPS_NO_OPTIMIZE setting.
    jax.clear_caches()
    args = jax.device_put(config.make_args(), MPS_DEVICE)
    result = jax.jit(config.func)(*args)
    jax.block_until_ready(result)
    return result


def test_fusion(config: FusionTestConfig, tmp_path: Path, monkeypatch):
    # 1. Fused path: dump post-pass IR and capture result.
    fused = _run_mps(
        config, dump_dir=tmp_path, disable_fusions=False, monkeypatch=monkeypatch
    )

    # 2. IR-level check: the dumped modules must contain the expected
    #    @mps.* custom_calls. A count of 0 is an explicit absence assertion.
    counts = _count_custom_calls(tmp_path)
    for name, expected in config.expected_custom_calls.items():
        actual = counts.get(name, 0)
        if expected == 0:
            assert actual == 0, (
                f"[{config.name}] expected no {name} fusion, found {actual}.\n"
                f"All custom_calls: {counts}"
            )
        else:
            assert actual >= expected, (
                f"[{config.name}] expected >= {expected} {name} fusion(s), "
                f"found {actual}.\nAll custom_calls: {counts}"
            )

    # 3. Unfused MPS path (JAX_MPS_NO_OPTIMIZE=1). Tight tolerance — same
    #    hardware, only the fused kernel differs. Catches regressions
    #    introduced specifically by the fusion.
    unfused = _run_mps(
        config, dump_dir=None, disable_fusions=True, monkeypatch=monkeypatch
    )
    numpy.testing.assert_allclose(
        numpy.asarray(fused),
        numpy.asarray(unfused),
        atol=config.fusion_atol,
        rtol=config.fusion_rtol,
        err_msg=f"[{config.name}] fused vs unfused MPS result mismatch",
    )

    # 4. CPU reference. Looser tolerance — different hardware/kernel paths.
    cpu_args = jax.device_put(config.make_args(), jax.devices("cpu")[0])
    cpu_result = jax.jit(config.func)(*cpu_args)
    numpy.testing.assert_allclose(
        numpy.asarray(fused),
        numpy.asarray(cpu_result),
        atol=config.atol,
        rtol=config.rtol,
        err_msg=f"[{config.name}] fused MPS vs CPU reference mismatch",
    )
