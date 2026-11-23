"""User-defined Metal kernels (mps.metal_kernel_jit / mps.metal_kernel_lib).

A raw Metal kernel has no CPU lowering, so each config runs the kernel on MPS
and an equivalent jnp reference elsewhere via ``lax.platform_dependent``; the
harness then compares the MPS result against the CPU reference as for any op.
Kernels have no VJP, so gradients are not tested.
"""

import functools
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from jax import lax, random
from jax import numpy as jnp

from jax_plugins.mps.ops import metal_kernel_jit, metal_kernel_lib

from .util import OperationTestConfig

_N = 256

_LIB_SOURCE = textwrap.dedent(
    """
    #include <metal_stdlib>
    using namespace metal;
    kernel void axpy(device const float* x   [[buffer(0)]],
                     device const float* y   [[buffer(1)]],
                     device float*       out [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
        out[i] = 2.0f * x[i] + y[i];
    }
    """
).strip()


@functools.cache
def _metallib_path() -> str:
    """Compile the config's .metallib once per process (needs Xcode's xcrun)."""
    assert shutil.which("xcrun"), "xcrun (Xcode Command Line Tools) is required"
    d = Path(tempfile.mkdtemp(prefix="jax_mps_metallib_"))
    src, air, lib = d / "k.metal", d / "k.air", d / "k.metallib"
    src.write_text(_LIB_SOURCE)
    subprocess.run(
        ["xcrun", "metal", "-O2", "-c", str(src), "-o", str(air)], check=True
    )
    subprocess.run(["xcrun", "metallib", str(air), "-o", str(lib)], check=True)
    return str(lib)


def _jit_fma(a, b, c):
    def on_mps(a, b, c):
        (out,) = metal_kernel_jit(
            "fma",
            [a, b, c],
            output_shapes=[a.shape],
            output_dtypes=[a.dtype],
            grid=(a.size, 1, 1),
            threadgroup=(64, 1, 1),
            # Compute in float32 and round once to the output dtype, so float16
            # results don't depend on whether the multiply-add is fused.
            source=f"uint i = thread_position_in_grid.x; if (i < {a.size}u) "
            f"out[i] = float(a[i]) * float(b[i]) + float(c[i]);",
            input_names=["a", "b", "c"],
            output_names=["out"],
        )
        return out

    def reference(a, b, c):
        f32 = jnp.float32
        return (a.astype(f32) * b.astype(f32) + c.astype(f32)).astype(a.dtype)

    return lax.platform_dependent(a, b, c, mps=on_mps, default=reference)


def _jit_sum_and_product(a, b):
    def on_mps(a, b):
        return tuple(
            metal_kernel_jit(
                "sum_and_product",
                [a, b],
                output_shapes=[a.shape, a.shape],
                output_dtypes=[a.dtype, a.dtype],
                grid=(a.size, 1, 1),
                threadgroup=(64, 1, 1),
                source="uint i = thread_position_in_grid.x; "
                "s[i] = a[i] + b[i]; p[i] = a[i] * b[i];",
                input_names=["a", "b"],
                output_names=["s", "p"],
            )
        )

    return lax.platform_dependent(a, b, mps=on_mps, default=lambda a, b: (a + b, a * b))


def _lib_axpy(x, y):
    def on_mps(x, y):
        (out,) = metal_kernel_lib(
            "axpy",
            [x, y],
            metallib_path=_metallib_path(),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
            grid=(x.size, 1, 1),
            threadgroup=(64, 1, 1),
        )
        return out

    return lax.platform_dependent(x, y, mps=on_mps, default=lambda x, y: 2 * x + y)


def make_metal_kernel_op_configs():
    with OperationTestConfig.module_name("metal_kernel"):
        yield OperationTestConfig(
            _jit_fma,
            lambda key: random.normal(key, (_N,)),
            lambda key: random.normal(key, (_N,)),
            lambda key: random.normal(key, (_N,)),
            differentiable_argnums=(),
            name="jit_fma",
        )
        yield OperationTestConfig(
            _jit_sum_and_product,
            lambda key: random.normal(key, (_N,)),
            lambda key: random.normal(key, (_N,)),
            differentiable_argnums=(),
            name="jit_multiple_outputs",
        )
        yield OperationTestConfig(
            _jit_fma,
            lambda key: random.normal(key, (_N,), dtype=jnp.float16),
            lambda key: random.normal(key, (_N,), dtype=jnp.float16),
            lambda key: random.normal(key, (_N,), dtype=jnp.float16),
            differentiable_argnums=(),
            name="jit_fma_float16",
        )
        yield OperationTestConfig(
            _lib_axpy,
            lambda key: random.normal(key, (_N,)),
            lambda key: random.normal(key, (_N,)),
            differentiable_argnums=(),
            name="lib_axpy",
        )
