"""Tests for the user-defined Metal kernel hooks.

Tests are MPS-only and check the output against a NumPy reference,
since raw user kernels have no host/CPU equivalent.
"""

import shutil
import subprocess
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_plugins.mps.ops import metal_kernel_jit, metal_kernel_lib

# The shared fixture skips when no MPS device exists and under JAX_TEST_MODE=cpu.
pytestmark = pytest.mark.usefixtures("mps_device")


def _run_on_mps(fn, *arrays):
    device = jax.devices("mps")[0]
    arrays = [jax.device_put(a, device) for a in arrays]
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


def test_metal_kernel_jit_same_name_different_source():
    """Two jitted kernels that share a name but differ in source must not alias."""
    n = 256

    def one(a, name, expr):
        (out,) = metal_kernel_jit(
            name,
            [a],
            output_shapes=[(n,)],
            output_dtypes=[jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(64, 1, 1),
            source=f"uint i = thread_position_in_grid.x; if (i < {n}u) out[i] = {expr};",
            input_names=["a"],
            output_names=["out"],
        )
        return out

    def fn(a):
        # Same name "k", different source, in a single jitted program.
        return one(a, "k", "a[i] + 1.0f"), one(a, "k", "a[i] * 2.0f")

    a = jax.random.normal(jax.random.PRNGKey(0), (n,))
    plus, times = _run_on_mps(fn, a)
    np.testing.assert_allclose(
        np.asarray(plus), np.asarray(a) + 1.0, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(times), np.asarray(a) * 2.0, rtol=1e-6, atol=1e-6
    )


def test_metal_kernel_jit_same_name_and_source_different_dtypes():
    """One name and source launched with float32 and float16 must each get a
    pipeline compiled for its own dtype signature."""
    n = 256

    def one(a, dtype):
        (out,) = metal_kernel_jit(
            "dtype_generic",
            [a],
            output_shapes=[(n,)],
            output_dtypes=[dtype],
            grid=(n, 1, 1),
            threadgroup=(64, 1, 1),
            source=f"uint i = thread_position_in_grid.x; if (i < {n}u) out[i] = a[i] * 2;",
            input_names=["a"],
            output_names=["out"],
        )
        return out

    def fn(a):
        return one(a, jnp.float32), one(a.astype(jnp.float16), jnp.float16)

    a = jax.random.normal(jax.random.PRNGKey(0), (n,))
    f32, f16 = _run_on_mps(fn, a)
    assert f16.dtype == jnp.float16
    np.testing.assert_allclose(
        np.asarray(f32), np.asarray(a) * 2.0, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(f16, dtype=np.float32),
        np.asarray(a).astype(np.float16).astype(np.float32) * 2.0,
        rtol=1e-3,
        atol=1e-3,
    )


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
    np.testing.assert_allclose(
        np.asarray(s), np.asarray(a) + np.asarray(b), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(p), np.asarray(a) * np.asarray(b), rtol=1e-6, atol=1e-6
    )


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
    kernel void vsub(device const float* a   [[buffer(0)]],
                     device const float* b   [[buffer(1)]],
                     device float*       out [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
        out[i] = a[i] - b[i];
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


def test_metal_kernel_lib_same_hash_name_different_kernels(vadd_metallib):
    """Two kernels from one library sharing a hash_name must not share a cached
    pipeline."""
    n = 256

    def one(name, a, b):
        (out,) = metal_kernel_lib(
            name,
            [a, b],
            metallib_path=vadd_metallib,
            output_shapes=[(n,)],
            output_dtypes=[jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(64, 1, 1),
            hash_name="shared",
        )
        return out

    def fn(a, b):
        return one("vadd", a, b), one("vsub", a, b)

    key = jax.random.PRNGKey(7)
    a, b = (jax.random.normal(k, (n,)) for k in jax.random.split(key, 2))
    added, subtracted = _run_on_mps(fn, a, b)
    np.testing.assert_allclose(
        np.asarray(added), np.asarray(a) + np.asarray(b), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(subtracted), np.asarray(a) - np.asarray(b), rtol=1e-6, atol=1e-6
    )


def test_metal_kernel_lib_non_contiguous_inputs(vadd_metallib):
    """Transposed and strided inputs are copied row-contiguous before the kernel
    reads them with flat indices."""
    rows, cols = 8, 16
    n = rows * cols

    def fn(x, y):
        (out,) = metal_kernel_lib(
            "vadd",
            [x.T, y[:, ::2]],
            metallib_path=vadd_metallib,
            output_shapes=[(n,)],
            output_dtypes=[jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(64, 1, 1),
        )
        return out

    key = jax.random.PRNGKey(8)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (cols, rows))
    y = jax.random.normal(ky, (rows, 2 * cols))
    out = np.asarray(_run_on_mps(fn, x, y))
    expected = (np.asarray(x).T + np.asarray(y)[:, ::2]).reshape(-1)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


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
    np.testing.assert_allclose(
        np.asarray(s), np.asarray(a) + np.asarray(b), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(p), np.asarray(a) * np.asarray(b), rtol=1e-6, atol=1e-6
    )


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


def test_metal_kernel_lib_function_constants_without_hash_name(vadd_metallib):
    """Two specializations of one kernel without a hash_name must not share a
    cached pipeline."""
    import struct

    n = 256
    params = struct.pack("<f", 1.0)

    def one(a, add_bias):
        (out,) = metal_kernel_lib(
            "scaled",
            [a],
            metallib_path=vadd_metallib,
            output_shapes=[(n,)],
            output_dtypes=[jnp.float32],
            grid=(n, 1, 1),
            threadgroup=(64, 1, 1),
            buffers=[
                {"slot": 0, "input": 0},
                {"slot": 2, "bytes": params},
                {"slot": 5, "output": 0},
            ],
            function_constants=[{"index": 7, "type": "bool", "value": add_bias}],
        )
        return out

    def fn(a):
        return one(a, False), one(a, True)

    a = jax.random.normal(jax.random.PRNGKey(6), (n,))
    plain, biased = _run_on_mps(fn, a)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(a), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(biased), np.asarray(a) + 1.0, rtol=1e-6, atol=1e-5
    )


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
            grid=(n, 1, 1),  # n threadgroups
            threadgroup=(1, 1, 1),
            dispatch="threadgroups",
        )
        return out

    a = jax.random.normal(jax.random.PRNGKey(5), (n,))
    out = np.asarray(_run_on_mps(fn, a))
    np.testing.assert_allclose(out, np.asarray(a) * 2.0, rtol=1e-6, atol=1e-6)


def test_metal_kernel_lib_rejects_duplicate_slot():
    """Two bindings on the same slot would silently last-win; reject up front."""
    a = jnp.zeros((4,), jnp.float32)
    with pytest.raises(ValueError, match="slot 0 bound more than once"):
        metal_kernel_lib(
            "k",
            [a],
            metallib_path="unused.metallib",
            output_shapes=[(4,)],
            output_dtypes=[jnp.float32],
            grid=(4, 1, 1),
            threadgroup=(1, 1, 1),
            buffers=[{"slot": 0, "input": 0}, {"slot": 0, "output": 0}],
        )


def test_metal_kernel_lib_rejects_too_many_positional_buffers_with_empty_list():
    """buffers=[] means positional binding, so it must hit the same slot limit
    as buffers=None."""
    inputs = [jnp.zeros((4,), jnp.float32)] * 31
    with pytest.raises(ValueError, match="buffer slots Metal provides"):
        metal_kernel_lib(
            "k",
            inputs,
            metallib_path="unused.metallib",
            output_shapes=[(4,)],
            output_dtypes=[jnp.float32],
            grid=(4, 1, 1),
            threadgroup=(1, 1, 1),
            buffers=[],
        )


def test_metal_kernel_lib_rejects_duplicate_constant_index():
    """A function-constant index set twice is ambiguous; reject up front."""
    a = jnp.zeros((4,), jnp.float32)
    with pytest.raises(ValueError, match="index 7 set more than once"):
        metal_kernel_lib(
            "k",
            [a],
            metallib_path="unused.metallib",
            output_shapes=[(4,)],
            output_dtypes=[jnp.float32],
            grid=(4, 1, 1),
            threadgroup=(1, 1, 1),
            function_constants=[
                {"index": 7, "type": "bool", "value": True},
                {"index": 7, "type": "int", "value": 3},
            ],
        )


def _bind_metal_kernel_lib_unchecked(a, metallib_path, *, buffers, function_constants):
    """Bind the primitive directly, bypassing the Python canonicalizers, so the
    backend's own validation is exercised."""
    from jax_plugins.mps.ops import _metal_kernel_lib_p

    (out,) = _metal_kernel_lib_p.bind(
        a,
        name="scaled",
        metallib_path=metallib_path,
        hash_name="",
        grid=(4, 1, 1),
        threadgroup=(1, 1, 1),
        dispatch="threads",
        buffers=buffers,
        function_constants=function_constants,
        out_shapes=((4,),),
        out_dtypes=(jnp.dtype(jnp.float32),),
    )
    return out


def test_metal_kernel_lib_backend_rejects_duplicate_slot(vadd_metallib):
    """The backend must reject a slot bound twice even when the Python
    canonicalizer is bypassed, instead of letting the later binding win."""

    def fn(a):
        return _bind_metal_kernel_lib_unchecked(
            a,
            vadd_metallib,
            buffers=((0, "input", 0, None), (0, "output", 0, None)),
            function_constants=None,
        )

    with pytest.raises(RuntimeError, match="slot out of range or bound more than once"):
        np.asarray(_run_on_mps(fn, jnp.zeros((4,), jnp.float32)))


def test_metal_kernel_lib_backend_rejects_duplicate_constant_index(vadd_metallib):
    """The backend must reject a function-constant index set twice even when the
    Python canonicalizer is bypassed."""

    def fn(a):
        return _bind_metal_kernel_lib_unchecked(
            a,
            vadd_metallib,
            buffers=((0, "input", 0, None), (5, "output", 0, None)),
            function_constants=((7, "bool", True), (7, "bool", False)),
        )

    with pytest.raises(RuntimeError, match="index negative or set more than once"):
        np.asarray(_run_on_mps(fn, jnp.zeros((4,), jnp.float32)))
