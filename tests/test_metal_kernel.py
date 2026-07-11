"""Tests for the user-defined Metal kernel hooks.

Tests are MPS-only and check the output against a NumPy reference,
since raw user kernels have no host/CPU equivalent.
"""

import shutil
import subprocess
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_plugins.mps.ops import metal_kernel_jit, metal_kernel_lib

try:
    MPS_DEVICE = jax.devices("mps")[0]
except (RuntimeError, IndexError):
    MPS_DEVICE = None

pytestmark = pytest.mark.skipif(MPS_DEVICE is None, reason="requires the MPS backend")


def _run_on_mps(fn, *arrays):
    arrays = [jax.device_put(a, MPS_DEVICE) for a in arrays]
    return jax.jit(fn)(*arrays)


def test_metal_kernel_jit_fused_multiply_add():
    """The a*b+c fused elementwise kernel, single output."""
    n = 1024
    source = f"""
        uint i = thread_position_in_grid.x;
        if (i < {n}u) out[i] = a[i] * b[i] + c[i];
    """

    def fn(a, b, c):
        (out,) = metal_kernel_jit(
            "fma",
            [a, b, c],
            output_shapes=[(n,)],
            output_dtypes=[jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
            source=source,
            input_names=["a", "b", "c"],
            output_names=["out"],
        )
        return out

    key = jax.random.PRNGKey(0)
    a, b, c = (jax.random.normal(k, (n,)) for k in jax.random.split(key, 3))
    out = np.asarray(_run_on_mps(fn, a, b, c))
    expected = np.asarray(a) * np.asarray(b) + np.asarray(c)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_metal_kernel_jit_multiple_outputs():
    """A kernel producing two outputs (sum and product) in one launch."""
    n = 512
    source = f"""
        uint i = thread_position_in_grid.x;
        if (i < {n}u) {{
            s[i] = a[i] + b[i];
            p[i] = a[i] * b[i];
        }}
    """

    def fn(a, b):
        return metal_kernel_jit(
            "sum_prod",
            [a, b],
            output_shapes=[(n,), (n,)],
            output_dtypes=[jnp.float32, jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(128, 1, 1),
            source=source,
            input_names=["a", "b"],
            output_names=["s", "p"],
        )

    key = jax.random.PRNGKey(1)
    a, b = (jax.random.normal(k, (n,)) for k in jax.random.split(key, 2))
    s, p = _run_on_mps(fn, a, b)
    np.testing.assert_allclose(np.asarray(s), np.asarray(a) + np.asarray(b), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(p), np.asarray(a) * np.asarray(b), rtol=1e-6, atol=1e-6)


_LIB_SOURCE = textwrap.dedent(
    """
    #include <metal_stdlib>
    using namespace metal;
    // Buffers bind positionally: inputs first (a, b), then outputs (out).
    kernel void vadd(device const float* a   [[buffer(0)]],
                     device const float* b   [[buffer(1)]],
                     device float*       out [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
        out[i] = a[i] + b[i];
    }
    kernel void vmul2(device const float* a   [[buffer(0)]],
                      device const float* b   [[buffer(1)]],
                      device float*       s   [[buffer(2)]],
                      device float*       p   [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
        s[i] = a[i] + b[i];
        p[i] = a[i] * b[i];
    }
    // Exercises explicit buffer slots, a set_bytes params blob, and a function
    // constant: out = a*scale (+1 if add_bias). Buffers are placed by the caller
    // at slots 0 (input), 2 (params bytes), 5 (output) — deliberately non-dense.
    struct ScaleParams { float scale; };
    constant bool add_bias [[function_constant(7)]];
    kernel void scaled(device const float* a       [[buffer(0)]],
                       constant ScaleParams& p     [[buffer(2)]],
                       device float*        out     [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
        float r = a[i] * p.scale;
        if (add_bias) r += 1.0f;
        out[i] = r;
    }
    // Indexes by threadgroup, so it must be launched with dispatch="threadgroups".
    kernel void tg_double(device const float* a   [[buffer(0)]],
                          device float*       out [[buffer(1)]],
                          uint g [[threadgroup_position_in_grid]]) {
        out[g] = a[g] * 2.0f;
    }
    """
).strip()


@pytest.fixture(scope="module")
def vadd_metallib(tmp_path_factory):
    """Compile a small .metallib for metal_kernel_lib tests."""
    if shutil.which("xcrun") is None:
        pytest.skip("requires xcrun (Xcode Command Line Tools) to build a .metallib")
    d = tmp_path_factory.mktemp("metallib")
    src, air, lib = d / "k.metal", d / "k.air", d / "k.metallib"
    src.write_text(_LIB_SOURCE)
    subprocess.run(
        ["xcrun", "metal", "-O2", "-c", str(src), "-o", str(air)],
        check=True,
    )
    subprocess.run(["xcrun", "metallib", str(air), "-o", str(lib)], check=True)
    return str(lib)


def test_metal_kernel_lib_single_output(vadd_metallib):
    n = 1024

    def fn(a, b):
        (out,) = metal_kernel_lib(
            "vadd",
            [a, b],
            metallib_path=vadd_metallib,
            output_shapes=[(n,)],
            output_dtypes=[jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
        )
        return out

    key = jax.random.PRNGKey(2)
    a, b = (jax.random.normal(k, (n,)) for k in jax.random.split(key, 2))
    out = np.asarray(_run_on_mps(fn, a, b))
    np.testing.assert_allclose(out, np.asarray(a) + np.asarray(b), rtol=1e-6, atol=1e-6)


def test_metal_kernel_lib_multiple_outputs(vadd_metallib):
    n = 512

    def fn(a, b):
        return metal_kernel_lib(
            "vmul2",
            [a, b],
            metallib_path=vadd_metallib,
            output_shapes=[(n,), (n,)],
            output_dtypes=[jnp.float32, jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(128, 1, 1),
        )

    key = jax.random.PRNGKey(3)
    a, b = (jax.random.normal(k, (n,)) for k in jax.random.split(key, 2))
    s, p = _run_on_mps(fn, a, b)
    np.testing.assert_allclose(np.asarray(s), np.asarray(a) + np.asarray(b), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(p), np.asarray(a) * np.asarray(b), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("add_bias", [False, True])
def test_metal_kernel_lib_buffers_and_function_constant(vadd_metallib, add_bias):
    """Explicit buffer slots, a set_bytes params blob, and a
    function constant that specializes the kernel."""
    import struct

    n = 256
    scale = 3.5
    params = struct.pack("<f", scale)  # struct ScaleParams { float scale; }

    def fn(a):
        (out,) = metal_kernel_lib(
            "scaled",
            [a],
            metallib_path=vadd_metallib,
            output_shapes=[(n,)],
            output_dtypes=[jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(64, 1, 1),
            hash_name=f"scaled_bias{int(add_bias)}",
            buffers=[
                {"slot": 0, "input": 0},
                {"slot": 2, "bytes": params},
                {"slot": 5, "output": 0},
            ],
            function_constants=[{"index": 7, "type": "bool", "value": add_bias}],
        )
        return out

    a = jax.random.normal(jax.random.PRNGKey(4), (n,))
    out = np.asarray(_run_on_mps(fn, a))
    expected = np.asarray(a) * scale + (1.0 if add_bias else 0.0)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-5)


def test_metal_kernel_lib_dispatch_threadgroups(vadd_metallib):
    """dispatch='threadgroups': grid is the threadgroup count, and the kernel
    indexes by threadgroup_position_in_grid."""
    n = 200

    def fn(a):
        (out,) = metal_kernel_lib(
            "tg_double",
            [a],
            metallib_path=vadd_metallib,
            output_shapes=[(n,)],
            output_dtypes=[jnp.float32],
            grid=(n, 1, 1),          # n threadgroups
            threadgroup=(1, 1, 1),
            dispatch="threadgroups",
        )
        return out

    a = jax.random.normal(jax.random.PRNGKey(5), (n,))
    out = np.asarray(_run_on_mps(fn, a))
    np.testing.assert_allclose(out, np.asarray(a) * 2.0, rtol=1e-6, atol=1e-6)
