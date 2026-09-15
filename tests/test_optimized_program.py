"""Tests for PJRT_Executable_OptimizedProgram.

jax-mps applies StableHLO simplification plus `MpsFusionPass` before
execution (`stablehlo_parser.cc`). `hlo_modules()` is the Python route into
this API: XLA's `PjRtCApiExecutable::GetHloModules()` calls
`PJRT_Executable_OptimizedProgram` and converts the MLIR it gets back into an
`HloModule`. So asserting on `hlo_modules()` output is a direct test of the
plugin's implementation.

Each test below pairs an assertion about the *output* with one about the
*input*.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def _optimized_hlo(device, fn, *args) -> str:
    """Compile `fn` on MPS and return the optimized HLO text.

    Goes through hlo_modules(), which is backed by
    PJRT_Executable_OptimizedProgram in the plugin. `device` is pinned
    explicitly via jax.default_device rather than relying on jax's
    default-backend priority ordering: under JAX_PLATFORMS=cpu (as CI runs
    the full suite, since GitHub's hosted macOS ARM64 runners have no Metal
    GPU) mps never registers, so an unpinned jax.jit would silently compile
    on CPU and every assertion here would compare against CPU XLA HLO instead
    of the plugin's output.
    """
    with jax.default_device(device):
        lowered = jax.jit(fn).lower(*args)
        compiled = lowered.compile()
        modules = compiled.runtime_executable().hlo_modules()
    assert len(modules) == 1, f"expected a single module, got {len(modules)}"
    return modules[0].to_string()


def test_optimized_program_is_available(mps_device):
    """The API is implemented at all (used to return UNIMPLEMENTED)."""
    text = _optimized_hlo(mps_device, lambda x: x + 1.0, jnp.ones((4,), jnp.float32))
    assert "HloModule" in text
    assert "ENTRY" in text


def test_optimized_program_reflects_simplification(mps_device):
    """Algebraic simplification must be visible in the reported program.

    `x * 1.0` is emitted by JAX but folded away by the plugin's simplification
    pass. If OptimizedProgram returned the pre-pass program, the multiply would
    still be there.
    """
    x = jnp.ones((4,), jnp.float32)

    # Guard against vacuity: JAX must actually emit the multiply for this to
    # be testing anything.
    stablehlo_in = jax.jit(lambda v: v * 1.0 + 2.0).lower(x).as_text()
    assert "stablehlo.multiply" in stablehlo_in, (
        "JAX no longer emits the multiply-by-one this test relies on"
    )

    text = _optimized_hlo(mps_device, lambda v: v * 1.0 + 2.0, x)
    assert " multiply(" not in text, (
        "multiply-by-one survived: OptimizedProgram is reporting pre-pass IR"
    )
    assert " add(" in text, "the add should still be present"


def test_optimized_program_reflects_mps_fusion(mps_device):
    """`@mps.*` fusions must be visible in the reported program.

    A `mps.addmm` custom call can only come from `MpsFusionPass`, so this
    cannot pass unless the post-pass module is what reaches the caller.
    """
    args = (
        jnp.ones((4, 8), jnp.float32),
        jnp.ones((8, 16), jnp.float32),
        jnp.ones((16,), jnp.float32),
    )

    def fn(a, b, bias):
        return jnp.dot(a, b) + bias

    stablehlo_in = jax.jit(fn).lower(*args).as_text()
    assert "dot_general" in stablehlo_in, "expected a dot_general to fuse"
    assert "mps.addmm" not in stablehlo_in, "input already fused"

    text = _optimized_hlo(mps_device, fn, *args)
    assert 'custom_call_target="mps.addmm"' in text, (
        "no mps.addmm: fusion output is not reaching the caller"
    )
    assert " dot(" not in text, "dot survived alongside the fusion"


def test_optimized_program_is_stable_across_calls(mps_device):
    """PJRT calls this twice (size query, then fill); both must agree.

    A mismatch between the two would overflow the caller's buffer, so the
    plugin caches the printed module. Calling repeatedly must be idempotent.
    """
    x = jnp.ones((8,), jnp.float32)
    fn = lambda v: jnp.tanh(v) * 2.0  # noqa: E731
    with jax.default_device(mps_device):
        compiled = jax.jit(fn).lower(x).compile()

        first = compiled.runtime_executable().hlo_modules()[0].to_string()
        second = compiled.runtime_executable().hlo_modules()[0].to_string()
    assert first == second


def test_optimized_program_survives_execution(mps_device):
    """Querying the program before and after running must not change it.

    Execute() walks the same module this API prints, so a handler mutating the
    IR would show up here.
    """
    x = jnp.ones((4,), jnp.float32)
    with jax.default_device(mps_device):
        compiled = jax.jit(lambda v: v * 3.0 + 1.0).lower(x).compile()

        before = compiled.runtime_executable().hlo_modules()[0].to_string()
        result = compiled(x)
        jax.block_until_ready(result)
        after = compiled.runtime_executable().hlo_modules()[0].to_string()

    assert before == after


@pytest.mark.parametrize(
    "fn,args",
    [
        (lambda x: jnp.sum(x), (jnp.ones((4, 4), jnp.float32),)),
        (lambda x: jnp.exp(x), (jnp.ones((3,), jnp.float32),)),
        (lambda x, y: jnp.concatenate([x, y]), (jnp.ones((2,), jnp.float32),) * 2),
        (lambda x: jnp.where(x > 0, x, -x), (jnp.ones((5,), jnp.float32),)),
    ],
    ids=["sum", "exp", "concatenate", "where"],
)
def test_optimized_program_across_shapes(mps_device, fn, args):
    """The API must work for a range of programs, not just the happy path."""
    text = _optimized_hlo(mps_device, fn, *args)
    assert "ENTRY" in text
