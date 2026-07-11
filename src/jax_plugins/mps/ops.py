"""Fused MLX operations exposed as JAX primitives via custom_call.

These primitives emit stablehlo.custom_call ops that the MPS backend
intercepts and dispatches to mlx::core::fast:: fused Metal kernels.

On MPS, both forward and backward passes run as fused MLX kernels (the
backward uses mlx::core::vjp). On non-MPS platforms, fallback lowerings
decompose to standard JAX ops.
"""

# pyright: reportArgumentType=false, reportOptionalCall=false
# pyright: reportFunctionMemberAccess=false, reportCallIssue=false

import json

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lax import ann as lax_ann
from jax._src.lax import lax as lax_lax
from jax._src.lax import linalg as lax_linalg
from jax._src.lax import special as lax_special
from jax._src.lib import _jax  # pyright: ignore[reportPrivateImportUsage]
from jax._src.lib.mlir import ir  # pyright: ignore[reportPrivateImportUsage]
from jax._src.lib.mlir.dialects import hlo


def _aval_to_ir_type(aval: core.ShapedArray) -> ir.Type:
    """Construct an MLIR ranked-tensor type for a ShapedArray.

    Avoids ``mlir.aval_to_ir_type`` whose signature is unstable across jax
    patch releases (see issue #162); ``mlir.dtype_to_ir_type`` has been
    signature-stable.
    """
    return ir.RankedTensorType.get(aval.shape, mlir.dtype_to_ir_type(aval.dtype))


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention (mps.sdpa)
# ---------------------------------------------------------------------------
# Two primitives: mps.sdpa (with boolean mask operand) and mps.sdpa_causal.
# Unmasked attention is just the masked primitive with an all-True mask.

_sdpa_p = core.Primitive("mps.sdpa")
_sdpa_p.multiple_results = False


def _sdpa_abstract(q, k, v, mask, *, scale):
    return core.ShapedArray(q.shape, q.dtype)


_sdpa_p.def_abstract_eval(_sdpa_abstract)


def _sdpa_impl(q, k, v, mask, *, scale):
    """Pure JAX fallback for non-MPS platforms.

    A boolean mask gates attention (True = attend); a float mask is an additive
    bias added to the pre-softmax scores (e.g. relative-position / ALiBi).
    """
    attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
    if jnp.issubdtype(mask.dtype, jnp.floating):
        attn = attn + mask.astype(attn.dtype)
    else:
        attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.matmul(attn, v)


_sdpa_p.def_impl(_sdpa_impl)


def _sdpa_lowering(ctx, q, k, v, mask, *, scale):
    result_type = _aval_to_ir_type(ctx.avals_out[0])
    return mlir.custom_call(
        call_target_name="mps.sdpa",
        result_types=[result_type],
        operands=[q, k, v, mask],
        backend_config=f'{{"scale": {scale}}}',
    ).results


# Causal variant — uses MLX's mask_mode="causal" string, not an array.
_sdpa_causal_p = core.Primitive("mps.sdpa_causal")
_sdpa_causal_p.multiple_results = False
_sdpa_causal_p.def_abstract_eval(
    lambda q, k, v, *, scale: core.ShapedArray(q.shape, q.dtype)
)


def _sdpa_causal_impl(q, k, v, *, scale):
    """Pure JAX fallback with causal mask."""
    T, S = q.shape[-2], k.shape[-2]
    attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
    mask = jnp.triu(jnp.full((T, S), jnp.finfo(attn.dtype).min), k=1)
    attn = attn + mask
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.matmul(attn, v)


_sdpa_causal_p.def_impl(_sdpa_causal_impl)


def _sdpa_causal_lowering(ctx, q, k, v, *, scale):
    result_type = _aval_to_ir_type(ctx.avals_out[0])
    return mlir.custom_call(
        call_target_name="mps.sdpa_causal",
        result_types=[result_type],
        operands=[q, k, v],
        backend_config=f'{{"scale": {scale}}}',
    ).results


def sdpa(q, k, v, *, scale=None, is_causal=False, mask=None):
    """Scaled dot-product attention using fused MLX kernel on MPS.

    Args:
        q: Queries, shape (B, N, T, H).
        k: Keys, shape (B, N_kv, S, H).
        v: Values, shape (B, N_kv, S, H).
        scale: Attention scale factor. Defaults to 1/sqrt(H).
        is_causal: Whether to apply causal (lower-triangular) masking.
        mask: Optional mask, shape broadcastable to (B, N, T, S). A boolean mask
            gates attention (True = attend, False = ignore); a float mask is added
            to the pre-softmax scores as an additive bias (relative-position / ALiBi).

    Returns:
        Output of shape (B, N, T, H).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scale = float(scale)
    if is_causal:
        if mask is not None:
            raise ValueError(
                "sdpa does not support combining is_causal=True with an explicit mask"
            )
        return _sdpa_causal_with_grad(q, k, v, scale)
    if mask is None:
        mask = jnp.bool_(True)
    else:
        mask = jnp.asarray(mask)
        if mask.dtype != jnp.bool_ and not jnp.issubdtype(mask.dtype, jnp.floating):
            raise ValueError(
                "sdpa mask must be boolean (gating) or floating (additive bias), "
                f"got dtype {mask.dtype}"
            )
    return _sdpa_with_grad(q, k, v, mask, scale)


_sdpa_bwd_p = core.Primitive("mps.sdpa_bwd")
_sdpa_bwd_p.multiple_results = True
_sdpa_bwd_p.def_abstract_eval(
    lambda q, k, v, mask, g, *, scale: (
        core.ShapedArray(q.shape, q.dtype),
        core.ShapedArray(k.shape, k.dtype),
        core.ShapedArray(v.shape, v.dtype),
    )
)
_sdpa_bwd_p.def_impl(
    lambda q, k, v, mask, g, *, scale: jax.vjp(
        lambda q, k, v: _sdpa_impl(q, k, v, mask, scale=scale), q, k, v
    )[1](g)
)

_sdpa_causal_bwd_p = core.Primitive("mps.sdpa_causal_bwd")
_sdpa_causal_bwd_p.multiple_results = True
_sdpa_causal_bwd_p.def_abstract_eval(
    lambda q, k, v, g, *, scale: (
        core.ShapedArray(q.shape, q.dtype),
        core.ShapedArray(k.shape, k.dtype),
        core.ShapedArray(v.shape, v.dtype),
    )
)
_sdpa_causal_bwd_p.def_impl(
    lambda q, k, v, g, *, scale: jax.vjp(
        lambda q, k, v: _sdpa_causal_impl(q, k, v, scale=scale), q, k, v
    )[1](g)
)


def _sdpa_bwd_lowering(ctx, q, k, v, mask, g, *, scale):
    avals = ctx.avals_out
    return mlir.custom_call(
        call_target_name="mps.sdpa_bwd",
        result_types=[_aval_to_ir_type(a) for a in avals],
        operands=[q, k, v, mask, g],
        backend_config=f'{{"scale": {scale}}}',
    ).results


def _sdpa_causal_bwd_lowering(ctx, q, k, v, g, *, scale):
    avals = ctx.avals_out
    return mlir.custom_call(
        call_target_name="mps.sdpa_causal_bwd",
        result_types=[_aval_to_ir_type(a) for a in avals],
        operands=[q, k, v, g],
        backend_config=f'{{"scale": {scale}}}',
    ).results


def _sdpa_with_grad(q, k, v, mask, scale):
    @jax.custom_vjp
    def fwd(q, k, v, mask):
        return _sdpa_p.bind(q, k, v, mask, scale=scale)

    def fwd_rule(q, k, v, mask):
        return fwd(q, k, v, mask), (q, k, v, mask)

    def bwd_rule(res, g):
        q, k, v, mask = res
        dq, dk, dv = _sdpa_bwd_p.bind(q, k, v, mask, g, scale=scale)
        return dq, dk, dv, None

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(q, k, v, mask)


def _sdpa_causal_with_grad(q, k, v, scale):
    @jax.custom_vjp
    def fwd(q, k, v):
        return _sdpa_causal_p.bind(q, k, v, scale=scale)

    def fwd_rule(q, k, v):
        return fwd(q, k, v), (q, k, v)

    def bwd_rule(res, g):
        q, k, v = res
        return _sdpa_causal_bwd_p.bind(q, k, v, g, scale=scale)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(q, k, v)


# ---------------------------------------------------------------------------
# RMS Normalization (mps.rms_norm)
# ---------------------------------------------------------------------------

_rms_norm_p = core.Primitive("mps.rms_norm")
_rms_norm_p.multiple_results = False


def _rms_norm_abstract(x, weight, *, eps):
    return core.ShapedArray(x.shape, x.dtype)


_rms_norm_p.def_abstract_eval(_rms_norm_abstract)


def _rms_norm_impl(x, weight, *, eps):
    """Pure JAX fallback."""
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + eps)
    return x * weight


_rms_norm_p.def_impl(_rms_norm_impl)


def _rms_norm_lowering(ctx, x, weight, *, eps):
    result_type = _aval_to_ir_type(ctx.avals_out[0])
    return mlir.custom_call(
        call_target_name="mps.rms_norm",
        result_types=[result_type],
        operands=[x, weight],
        backend_config=f'{{"eps": {eps}}}',
    ).results


_rms_norm_bwd_p = core.Primitive("mps.rms_norm_bwd")
_rms_norm_bwd_p.multiple_results = True
_rms_norm_bwd_p.def_abstract_eval(
    lambda x, w, g, *, eps: (
        core.ShapedArray(x.shape, x.dtype),
        core.ShapedArray(w.shape, w.dtype),
    )
)
_rms_norm_bwd_p.def_impl(
    lambda x, w, g, *, eps: jax.vjp(lambda x, w: _rms_norm_impl(x, w, eps=eps), x, w)[
        1
    ](g)
)


def _rms_norm_bwd_lowering(ctx, x, w, g, *, eps):
    avals = ctx.avals_out
    return mlir.custom_call(
        call_target_name="mps.rms_norm_bwd",
        result_types=[_aval_to_ir_type(a) for a in avals],
        operands=[x, w, g],
        backend_config=f'{{"eps": {eps}}}',
    ).results


def rms_norm(x, weight, *, eps=1e-6):
    """RMS normalization using fused MLX kernel on MPS."""
    eps = float(eps)

    @jax.custom_vjp
    def fwd(x, weight):
        return _rms_norm_p.bind(x, weight, eps=eps)

    def fwd_rule(x, weight):
        return fwd(x, weight), (x, weight)

    def bwd_rule(res, g):
        x, weight = res
        return _rms_norm_bwd_p.bind(x, weight, g, eps=eps)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(x, weight)


# ---------------------------------------------------------------------------
# Layer Normalization (mps.layer_norm)
# ---------------------------------------------------------------------------

_layer_norm_p = core.Primitive("mps.layer_norm")
_layer_norm_p.multiple_results = False


def _layer_norm_abstract(x, weight, bias, *, eps):
    return core.ShapedArray(x.shape, x.dtype)


_layer_norm_p.def_abstract_eval(_layer_norm_abstract)


def _layer_norm_impl(x, weight, bias, *, eps):
    """Pure JAX fallback."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(variance + eps)
    return x * weight + bias


_layer_norm_p.def_impl(_layer_norm_impl)


def _layer_norm_lowering(ctx, x, weight, bias, *, eps):
    result_type = _aval_to_ir_type(ctx.avals_out[0])
    return mlir.custom_call(
        call_target_name="mps.layer_norm",
        result_types=[result_type],
        operands=[x, weight, bias],
        backend_config=f'{{"eps": {eps}}}',
    ).results


_layer_norm_bwd_p = core.Primitive("mps.layer_norm_bwd")
_layer_norm_bwd_p.multiple_results = True
_layer_norm_bwd_p.def_abstract_eval(
    lambda x, w, b, g, *, eps: (
        core.ShapedArray(x.shape, x.dtype),
        core.ShapedArray(w.shape, w.dtype),
        core.ShapedArray(b.shape, b.dtype),
    )
)
_layer_norm_bwd_p.def_impl(
    lambda x, w, b, g, *, eps: jax.vjp(
        lambda x, w, b: _layer_norm_impl(x, w, b, eps=eps), x, w, b
    )[1](g)
)


def _layer_norm_bwd_lowering(ctx, x, w, b, g, *, eps):
    avals = ctx.avals_out
    return mlir.custom_call(
        call_target_name="mps.layer_norm_bwd",
        result_types=[_aval_to_ir_type(a) for a in avals],
        operands=[x, w, b, g],
        backend_config=f'{{"eps": {eps}}}',
    ).results


def layer_norm(x, weight, bias, *, eps=1e-5):
    """Layer normalization using fused MLX kernel on MPS."""
    eps = float(eps)

    @jax.custom_vjp
    def fwd(x, weight, bias):
        return _layer_norm_p.bind(x, weight, bias, eps=eps)

    def fwd_rule(x, weight, bias):
        return fwd(x, weight, bias), (x, weight, bias)

    def bwd_rule(res, g):
        x, weight, bias = res
        return _layer_norm_bwd_p.bind(x, weight, bias, g, eps=eps)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(x, weight, bias)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (mps.rope)
# ---------------------------------------------------------------------------

_rope_p = core.Primitive("mps.rope")
_rope_p.multiple_results = False


def _rope_abstract(x, offset, *, dims, traditional, base, rope_scale):
    return core.ShapedArray(x.shape, x.dtype)


_rope_p.def_abstract_eval(_rope_abstract)


def _rope_impl(x, offset, *, dims, traditional, base, rope_scale):
    """Pure JAX fallback for RoPE."""
    if traditional:
        raise NotImplementedError(
            "mps.rope fallback only supports non-traditional (half-split) RoPE."
        )
    half_dim = dims // 2
    freqs = 1.0 / (base ** (jnp.arange(0, dims, 2, dtype=jnp.float32) / dims))
    positions = (jnp.arange(x.shape[-2], dtype=jnp.float32) + offset) * rope_scale
    angles = positions[:, None] * freqs[None, :]
    cos_half = jnp.cos(angles)
    sin_half = jnp.sin(angles)
    cos = jnp.concatenate([cos_half, cos_half], axis=-1)
    sin = jnp.concatenate([sin_half, sin_half], axis=-1)
    x1, x2 = x[..., :half_dim], x[..., half_dim:dims]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    result = (x[..., :dims] * cos + rotated * sin).astype(x.dtype)
    if dims < x.shape[-1]:
        result = jnp.concatenate([result, x[..., dims:]], axis=-1)
    return result


_rope_p.def_impl(_rope_impl)


def _rope_lowering(ctx, x, offset, *, dims, traditional, base, rope_scale):
    result_type = _aval_to_ir_type(ctx.avals_out[0])
    traditional_str = "true" if traditional else "false"
    return mlir.custom_call(
        call_target_name="mps.rope",
        result_types=[result_type],
        operands=[x, offset],
        backend_config=(
            f'{{"dims": {dims}, "traditional": {traditional_str}, '
            f'"base": {base}, "rope_scale": {rope_scale}}}'
        ),
    ).results


_rope_bwd_p = core.Primitive("mps.rope_bwd")
_rope_bwd_p.multiple_results = False
_rope_bwd_p.def_abstract_eval(
    lambda x, offset, g, *, dims, traditional, base, rope_scale: core.ShapedArray(
        x.shape, x.dtype
    )
)
_rope_bwd_p.def_impl(
    lambda x, offset, g, *, dims, traditional, base, rope_scale: jax.vjp(
        lambda x: _rope_impl(
            x,
            offset,
            dims=dims,
            traditional=traditional,
            base=base,
            rope_scale=rope_scale,
        ),
        x,
    )[1](g)[0]
)


def _rope_bwd_lowering(ctx, x, offset, g, *, dims, traditional, base, rope_scale):
    result_type = _aval_to_ir_type(ctx.avals_out[0])
    traditional_str = "true" if traditional else "false"
    return mlir.custom_call(
        call_target_name="mps.rope_bwd",
        result_types=[result_type],
        operands=[x, offset, g],
        backend_config=(
            f'{{"dims": {dims}, "traditional": {traditional_str}, '
            f'"base": {base}, "rope_scale": {rope_scale}}}'
        ),
    ).results


def rope(x, *, dims, base=10000.0, scale=1.0, offset=0, traditional=False):
    """Rotary position embeddings using fused MLX kernel on MPS."""
    dims = int(dims)
    if dims <= 0 or dims % 2 != 0:
        raise ValueError(f"dims must be a positive even integer, got {dims}")
    traditional = bool(traditional)
    base = float(base)
    rope_scale = float(scale)
    offset_arr = jnp.int32(offset)
    params = dict(
        dims=dims,
        traditional=traditional,
        base=base,
        rope_scale=rope_scale,
    )

    @jax.custom_vjp
    def fwd(x, offset_arr):
        return _rope_p.bind(x, offset_arr, **params)

    def fwd_rule(x, offset_arr):
        return fwd(x, offset_arr), (x, offset_arr)

    def bwd_rule(res, g):
        x, offset_arr = res
        return (_rope_bwd_p.bind(x, offset_arr, g, **params), None)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(x, offset_arr)


# ---------------------------------------------------------------------------
# GELU approximate (mps.gelu)
# ---------------------------------------------------------------------------

_gelu_p = core.Primitive("mps.gelu")
_gelu_p.multiple_results = False


def _gelu_abstract(x):
    return core.ShapedArray(x.shape, x.dtype)


_gelu_p.def_abstract_eval(_gelu_abstract)


def _gelu_impl_dispatch(x):
    """Dispatch to the original (unpatched) gelu, avoiding recursion."""
    fn = _gelu_original
    if fn is None:
        fn = jax.nn.gelu
    # Guard: if fn is our patched version, decompose manually to avoid recursion.
    if getattr(fn, "_mps_patched", False):
        sqrt_2_over_pi = jnp.array(0.7978845834732056, dtype=x.dtype)
        cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x**3))))
        return x * cdf
    return fn(x, approximate=True)


_gelu_p.def_impl(_gelu_impl_dispatch)


def _gelu_lowering(ctx, x):
    result_type = _aval_to_ir_type(ctx.avals_out[0])
    return mlir.custom_call(
        call_target_name="mps.gelu",
        result_types=[result_type],
        operands=[x],
        backend_config="",
    ).results


_gelu_bwd_p = core.Primitive("mps.gelu_bwd")
_gelu_bwd_p.multiple_results = False
_gelu_bwd_p.def_abstract_eval(lambda x, g: core.ShapedArray(x.shape, x.dtype))
_gelu_bwd_p.def_impl(lambda x, g: jax.vjp(lambda x: _gelu_impl_dispatch(x), x)[1](g)[0])


def _gelu_bwd_lowering(ctx, x, g):
    result_type = _aval_to_ir_type(ctx.avals_out[0])
    return mlir.custom_call(
        call_target_name="mps.gelu_bwd",
        result_types=[result_type],
        operands=[x, g],
        backend_config="",
    ).results


def gelu(x):
    """Approximate GELU using fused MLX kernel on MPS."""

    @jax.custom_vjp
    def fwd(x):
        return _gelu_p.bind(x)

    def fwd_rule(x):
        return fwd(x), (x,)

    def bwd_rule(res, g):
        (x,) = res
        return (_gelu_bwd_p.bind(x, g),)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(x)


# ---------------------------------------------------------------------------
# User-defined Metal kernels
# ---------------------------------------------------------------------------

_metal_kernel_jit_p = core.Primitive("mps.metal_kernel_jit")
_metal_kernel_jit_p.multiple_results = True


def _metal_kernel_jit_abstract(
    *inputs,
    name,
    source,
    header,
    input_names,
    output_names,
    grid,
    threadgroup,
    out_shapes,
    out_dtypes,
):
    return tuple(
        core.ShapedArray(shape, jnp.dtype(dtype))
        for shape, dtype in zip(out_shapes, out_dtypes)
    )


_metal_kernel_jit_p.def_abstract_eval(_metal_kernel_jit_abstract)


def _metal_kernel_jit_impl(*inputs, **params):
    raise NotImplementedError(
        "mps.metal_kernel_jit is only lowered on the MPS backend (no host/CPU "
        "equivalent for arbitrary Metal source)."
    )


_metal_kernel_jit_p.def_impl(_metal_kernel_jit_impl)


def _metal_kernel_jit_lowering(
    ctx,
    *inputs,
    name,
    source,
    header,
    input_names,
    output_names,
    grid,
    threadgroup,
    out_shapes,
    out_dtypes,
):
    cfg = {
        "name": name,
        "source": source,
        "header": header,
        "input_names": list(input_names),
        "output_names": list(output_names),
        "grid": [int(g) for g in grid],
        "threadgroup": [int(t) for t in threadgroup],
    }
    return mlir.custom_call(
        call_target_name="mps.metal_kernel_jit",
        result_types=[_aval_to_ir_type(a) for a in ctx.avals_out],
        operands=list(inputs),
        backend_config=json.dumps(cfg),
    ).results


# Metal exposes buffer slots [[buffer(0)]]..[[buffer(30)]].
_MAX_METAL_BUFFERS = 31


def _dim3(x, name, minimum):
    """Coerce a launch dimension to an (x, y, z) int tuple, rejecting other lengths."""
    dims = tuple(int(v) for v in x)
    if len(dims) != 3:
        raise ValueError(f"{name} must have 3 entries (x, y, z), got {len(dims)}")
    if any(d < minimum for d in dims):
        raise ValueError(f"{name} entries must be >= {minimum}, got {dims}")
    return dims


def metal_kernel_jit(
    name,
    inputs,
    *,
    output_shapes,
    output_dtypes,
    grid,
    threadgroup,
    source,
    header="",
    input_names=None,
    output_names=None,
):
    """JIT-compile and launch a Metal kernel from MSL source.

    `source` is the MSL kernel body; reference inputs/outputs by their names
    (input_names/output_names, default in0.. / out0..) as row-contiguous device
    pointers. `grid` is the total thread count per dim, `threadgroup` the threads per group.
    Outputs are allocated uninitialized by MLX from output_shapes/output_dtypes.
    """
    inputs = [jnp.asarray(x) for x in inputs]
    if input_names is None:
        input_names = tuple(f"in{i}" for i in range(len(inputs)))
    if len(input_names) != len(inputs):
        raise ValueError("input_names must match the number of inputs")

    if len(output_shapes) != len(output_dtypes):
        raise ValueError("output_shapes and output_dtypes must have the same length")

    if output_names is None:
        output_names = tuple(f"out{i}" for i in range(len(output_shapes)))
    if len(output_names) != len(output_shapes):
        raise ValueError("output_names must match the number of outputs")

    if len(inputs) + len(output_shapes) > _MAX_METAL_BUFFERS:
        raise ValueError(
            f"metal_kernel_jit: {len(inputs)} inputs + {len(output_shapes)} outputs "
            f"needs more than the {_MAX_METAL_BUFFERS} buffer slots Metal provides"
        )

    out_shapes = tuple(tuple(int(d) for d in s) for s in output_shapes)
    out_dtypes = tuple(jnp.dtype(d) for d in output_dtypes)
    return _metal_kernel_jit_p.bind(
        *inputs,
        name=str(name),
        source=str(source),
        header=str(header),
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        grid=_dim3(grid, "grid", 1),
        threadgroup=_dim3(threadgroup, "threadgroup", 1),
        out_shapes=out_shapes,
        out_dtypes=out_dtypes,
    )


_metal_kernel_lib_p = core.Primitive("mps.metal_kernel_lib")
_metal_kernel_lib_p.multiple_results = True


def _metal_kernel_lib_abstract(
    *inputs,
    name,
    metallib_path,
    hash_name,
    grid,
    threadgroup,
    dispatch,
    buffers,
    function_constants,
    out_shapes,
    out_dtypes,
):
    return tuple(
        core.ShapedArray(shape, jnp.dtype(dtype))
        for shape, dtype in zip(out_shapes, out_dtypes)
    )


_metal_kernel_lib_p.def_abstract_eval(_metal_kernel_lib_abstract)


def _metal_kernel_lib_impl(*inputs, **params):
    raise NotImplementedError(
        "mps.metal_kernel_lib is only lowered on the MPS backend."
    )


_metal_kernel_lib_p.def_impl(_metal_kernel_lib_impl)


def _metal_kernel_lib_lowering(
    ctx,
    *inputs,
    name,
    metallib_path,
    hash_name,
    grid,
    threadgroup,
    dispatch,
    buffers,
    function_constants,
    out_shapes,
    out_dtypes,
):
    cfg = {
        "name": name,
        "metallib_path": metallib_path,
        "grid": [int(g) for g in grid],
        "threadgroup": [int(t) for t in threadgroup],
    }
    if hash_name:
        cfg["hash_name"] = hash_name
    if dispatch != "threads":
        cfg["dispatch"] = dispatch
    if buffers is not None:
        cfg["buffers"] = [
            {"slot": slot, "kind": kind}
            | ({"bytes": list(payload)} if kind == "bytes" else {"arg": arg})
            for (slot, kind, arg, payload) in buffers
        ]
    if function_constants is not None:
        cfg["function_constants"] = [
            {"index": idx, "type": typ, "value": val}
            for (idx, typ, val) in function_constants
        ]
    return mlir.custom_call(
        call_target_name="mps.metal_kernel_lib",
        result_types=[_aval_to_ir_type(a) for a in ctx.avals_out],
        operands=list(inputs),
        backend_config=json.dumps(cfg),
    ).results


def _canon_buffers(buffers, n_inputs, n_outputs):
    """Normalize buffer specs to a hashable tuple of (slot, kind, arg, bytes)."""
    if not buffers:  # None or empty => default positional binding
        return None
    out = []
    seen_slots = set()
    for b in buffers:
        slot = int(b["slot"])
        if not 0 <= slot < _MAX_METAL_BUFFERS:
            raise ValueError(
                f"buffer slot must be in 0..{_MAX_METAL_BUFFERS - 1}, got {slot}"
            )
        if slot in seen_slots:
            raise ValueError(f"buffer slot {slot} bound more than once")
        seen_slots.add(slot)
        kinds = [k for k in ("input", "output", "bytes") if k in b]
        if len(kinds) != 1:
            raise ValueError(
                "buffer spec needs exactly one of 'input'/'output'/'bytes'"
            )
        (kind,) = kinds
        if kind == "input":
            arg = int(b["input"])
            if not 0 <= arg < n_inputs:
                raise ValueError(
                    f"buffer input index {arg} out of range (0..{n_inputs - 1})"
                )
            out.append((slot, "input", arg, None))
        elif kind == "output":
            arg = int(b["output"])
            if not 0 <= arg < n_outputs:
                raise ValueError(
                    f"buffer output index {arg} out of range (0..{n_outputs - 1})"
                )
            out.append((slot, "output", arg, None))
        else:  # bytes
            out.append((slot, "bytes", None, bytes(b["bytes"])))
    return tuple(out)


def _canon_function_constants(fcs):
    """Normalize function-constant specs to a hashable tuple."""
    if not fcs:  # None or empty => no specialization
        return None
    out = []
    seen_indices = set()
    for c in fcs:
        index = int(c["index"])
        if index < 0:
            raise ValueError(f"function_constant index must be >= 0, got {index}")
        if index in seen_indices:
            raise ValueError(f"function_constant index {index} set more than once")
        seen_indices.add(index)
        typ = str(c["type"])
        if typ not in ("bool", "int", "uint", "float"):
            raise ValueError(
                f"function_constant type must be bool/int/uint/float, got {typ!r}"
            )
        val = c["value"]
        if typ == "bool":
            val = bool(val)
        elif typ == "float":
            val = float(val)
        else:  # int or uint
            val = int(val)
            if typ == "uint" and val < 0:
                raise ValueError(
                    f"uint function_constant value must be non-negative, got {val}"
                )
        out.append((index, typ, val))
    return tuple(out)


def metal_kernel_lib(
    name,
    inputs,
    *,
    metallib_path,
    output_shapes,
    output_dtypes,
    grid,
    threadgroup,
    dispatch="threads",
    hash_name=None,
    buffers=None,
    function_constants=None,
):
    """Dispatch a named kernel from a precompiled .metallib.

    `name` is the kernel function's name inside the library at `metallib_path`.
    `grid` is the total thread count per dim, `threadgroup` the threads per group.
    Outputs are allocated uninitialized by MLX from output_shapes/output_dtypes.

    ``buffers=None``: operands bind to buffers 0..N-1 and outputs
    to N..N+M-1, all row-contiguous.

    Otherwise, pass `buffers` to place inputs/outputs/raw-bytes at explicit
    slots, and `function_constants` to specialize the pipeline.

    - `buffers`: list of dicts, each with ``"slot"`` and exactly one of
      ``"input": operand_idx``, ``"output": result_idx``, or ``"bytes": bytes``.
    - `function_constants`: list of dicts ``{"index", "type", "value"}`` where
      type is one of ``"bool" | "int" | "uint" | "float"``.
    - `hash_name`: pipeline cache key (defaults to `name`); give distinct keys to
      distinct function-constant specializations of the same kernel.
    - `dispatch`: ``"threads"`` (default; `grid` is the total thread count per
      dim) or ``"threadgroups"`` (`grid` is the threadgroup count per dim, for
      kernels that index by threadgroup_position_in_grid).
    """
    if dispatch not in ("threads", "threadgroups"):
        raise ValueError("dispatch must be 'threads' or 'threadgroups'")
    if len(output_shapes) != len(output_dtypes):
        raise ValueError("output_shapes and output_dtypes must have the same length")
    inputs = [jnp.asarray(x) for x in inputs]
    out_shapes = tuple(tuple(int(d) for d in s) for s in output_shapes)
    out_dtypes = tuple(jnp.dtype(d) for d in output_dtypes)
    if buffers is None and len(inputs) + len(out_shapes) > _MAX_METAL_BUFFERS:
        raise ValueError(
            f"metal_kernel_lib: {len(inputs)} inputs + {len(out_shapes)} outputs "
            f"needs more than the {_MAX_METAL_BUFFERS} buffer slots Metal provides"
        )
    return _metal_kernel_lib_p.bind(
        *inputs,
        name=str(name),
        metallib_path=str(metallib_path),
        hash_name=str(hash_name) if hash_name else "",
        grid=_dim3(grid, "grid", 1),
        threadgroup=_dim3(threadgroup, "threadgroup", 1),
        dispatch=str(dispatch),
        buffers=_canon_buffers(buffers, len(inputs), len(out_shapes)),
        function_constants=_canon_function_constants(function_constants),
        out_shapes=out_shapes,
        out_dtypes=out_dtypes,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Linalg lowerings: eigh, qr  (MPS-specific custom_call lowerings)
# ---------------------------------------------------------------------------


def _eigh_lowering(
    ctx, operand, *, lower, sort_eigenvalues, subset_by_index, algorithm
):
    """MPS lowering for eigh: emit a custom_call @mps.eigh."""
    operand_aval = ctx.avals_in[0]
    n = operand_aval.shape[-1]
    if subset_by_index is not None and subset_by_index != (0, n):
        raise NotImplementedError("subset_by_index not supported on MPS")
    if algorithm is not None:
        raise NotImplementedError("algorithm selection not supported on MPS")
    del sort_eigenvalues  # kernel always sorts ascending
    v_aval, w_aval = ctx.avals_out
    v_type = _aval_to_ir_type(v_aval)
    w_type = _aval_to_ir_type(w_aval)
    return mlir.custom_call(
        call_target_name="mps.eigh",
        result_types=[v_type, w_type],
        operands=[operand],
        backend_config=f'{{"lower": {str(lower).lower()}}}',
    ).results


def _qr_lowering(ctx, operand, *, full_matrices, **kwargs):
    """MPS lowering for qr: emit a custom_call @mps.qr."""
    del kwargs  # pivoting, use_magma — not applicable on MPS
    # Result types from ctx.avals_out already reflect full_matrices setting.
    q_aval, r_aval = ctx.avals_out
    q_type = _aval_to_ir_type(q_aval)
    r_type = _aval_to_ir_type(r_aval)
    return mlir.custom_call(
        call_target_name="mps.qr",
        result_types=[q_type, r_type],
        operands=[operand],
        backend_config=f'{{"full_matrices": {str(full_matrices).lower()}}}',
    ).results


def _svd_via_transpose(operand, *, full_matrices, compute_uv):
    """Compute SVD of a wide matrix (M < N) by reducing to the tall case.

    For A with shape (..., M, N) and M < N:  A = (Aᵀ)ᵀ = (Û Σ V̂ᵀ)ᵀ = V̂ Σ Ûᵀ,
    where Û Σ V̂ᵀ is the SVD of Aᵀ. So U = V̂ = (V̂ᵀ)ᵀ, Vᵀ = Ûᵀ.
    """
    operand_t = jnp.swapaxes(operand, -1, -2)
    if compute_uv:
        u_hat, s, vt_hat = lax_linalg.svd(
            operand_t, full_matrices=full_matrices, compute_uv=True
        )
        u = jnp.swapaxes(vt_hat, -1, -2)
        vt = jnp.swapaxes(u_hat, -1, -2)
        # svd_p abstract eval order is (s, u, vt) — match that here so
        # mlir.lower_fun lines up the results with ctx.avals_out.
        return s, u, vt
    return lax_linalg.svd(operand_t, full_matrices=full_matrices, compute_uv=False)


def _svd_lowering(
    ctx, operand, *, full_matrices, compute_uv, subset_by_index, algorithm=None
):
    """MPS lowering for svd: emit a custom_call @mps.svd."""
    if subset_by_index is not None:
        raise NotImplementedError("subset_by_index not supported on MPS")
    if algorithm is not None:
        raise NotImplementedError("algorithm selection not supported on MPS")
    operand_aval = ctx.avals_in[0]
    m, n = operand_aval.shape[-2], operand_aval.shape[-1]
    if m < n:
        # MPS SVD kernel only handles tall matrices (M >= N). Reduce to the
        # tall case by transposing; the recursive lax_linalg.svd call lowers
        # again through this rule with the dimensions swapped.
        return mlir.lower_fun(
            lambda x: _svd_via_transpose(
                x, full_matrices=full_matrices, compute_uv=compute_uv
            ),
            multiple_results=compute_uv,
        )(ctx, operand)
    fm = "true" if full_matrices else "false"
    if compute_uv:
        # JAX svd_p abstract eval returns (s, u, vt) – note s first!
        s_aval, u_aval, vt_aval = ctx.avals_out
        return mlir.custom_call(
            call_target_name="mps.svd",
            result_types=[
                _aval_to_ir_type(s_aval),
                _aval_to_ir_type(u_aval),
                _aval_to_ir_type(vt_aval),
            ],
            operands=[operand],
            backend_config=f'{{"compute_uv": true, "full_matrices": {fm}}}',
        ).results
    else:
        (s_aval,) = ctx.avals_out
        return mlir.custom_call(
            call_target_name="mps.svd",
            result_types=[_aval_to_ir_type(s_aval)],
            operands=[operand],
            backend_config=f'{{"compute_uv": false, "full_matrices": {fm}}}',
        ).results


def _make_native_unary_lowering(call_target_name):
    """Return a JAX MLIR lowering that emits a stablehlo.custom_call with the
    given target name (expected to be one of the mhlo.* entries recognized by
    our C++ HandleCustomCall). Keeps us off the CHLO decomposition path so the
    unary op stays a single MLX kernel call."""

    def _lowering(ctx, x):
        (aval_out,) = ctx.avals_out
        return mlir.custom_call(
            call_target_name=call_target_name,
            result_types=[_aval_to_ir_type(aval_out)],
            operands=[x],
            backend_config="",
        ).results

    return _lowering


def _logistic_lowering(ctx, x, *, accuracy=None):
    """MPS lowering for logistic_p: emit a native stablehlo.logistic op.

    Upstream JAX lowers logistic_p to the explicit 1/(1+exp(-x)) decomposition
    because the HLO LogisticOp lowering had numerical issues on some backends
    (see the commented-out registration in jax/_src/lax/lax.py). On MPS we route
    stablehlo.logistic to mlx::core::sigmoid, which is numerically stable, so we
    keep it as a single op instead of a four-op decomposition. The accuracy
    attribute is irrelevant to the MLX kernel and is dropped (matching
    logistic_impl).
    """
    del accuracy
    return [hlo.logistic(x)]


# ---------------------------------------------------------------------------
# Approximate top-k (approx_max_k / approx_min_k)
# ---------------------------------------------------------------------------
# JAX registers the fast approx_top_k lowering for the TPU platform only
# (jax/_src/lax/ann.py); every other backend gets the `is_fallback=True`
# lowering that computes exact top-k. We provide a real MPS lowering — the
# intended platform extension point, mirroring how TPU does it — that does a
# recall-driven strided reduction for large inputs (5-11x faster than exact at
# scale; see scratch benchmarks) and falls back to exact top-k where that does
# not pay (k==1, or input small enough to fit one reduction tile).

# Below this reduction-axis size the strided path's extra kernels (reshape,
# block argmax, gather, candidate top-k) cost more than a single argpartition
# over the whole axis, so we keep exact top-k. Measured crossover on M-class
# GPUs is ~8-16k (noisy); 16384 is the conservative point above which the
# approximate path is reliably faster. Exact below it is never slower than the
# status quo and still numerically valid (recall 1.0).
_APPROX_TOPK_MIN_REDUCTION_SIZE = 16384


def _approx_top_k_mps_impl(
    operand,
    *,
    k,
    reduction_dimension,
    recall_target,
    is_max_k,
    reduction_input_size_override,
    aggregate_to_topk,
):
    """Pure-JAX approximate top-k for MPS, decomposed into supported ops.

    Uses XLA's own reduction-size formula (``approx_top_k_reduction_output_size``)
    to choose the block count, so the achieved recall matches ``recall_target``
    and ``aggregate_to_topk=False`` output shapes line up with JAX's abstract
    eval. Returns ``(values, int32 indices)`` along ``reduction_dimension``.
    """
    rank = operand.ndim
    d = reduction_dimension % rank
    x = jnp.moveaxis(operand, d, -1)
    n = x.shape[-1]

    # (candidate count, log2 of block count) from XLA's recall model. Query with
    # aggregate_to_topk=False to get the intermediate reduction structure even
    # when we aggregate to exactly k afterwards.
    reduced_size, log2_red = _jax.approx_top_k_reduction_output_size(
        n, rank, k, float(recall_target), False, reduction_input_size_override
    )
    num_blocks = 1 << max(int(log2_red), 0)

    # Solve everything as a max-problem; negate for min-k (indices unaffected).
    work = x if is_max_k else -x

    def finalize(vals, idx):
        vals = vals if is_max_k else -vals
        return (
            jnp.moveaxis(vals, -1, d),
            jnp.moveaxis(idx.astype(jnp.int32), -1, d),
        )

    # Exact fast path: aggregate to exactly k with no beneficial reduction.
    # k==1 is already a single max-reduction; num_blocks<=1 means N fits one
    # tile; below the crossover size the strided path is slower than exact.
    # All are exact (recall 1.0) and at least as fast as the strided path.
    if aggregate_to_topk and (
        k == 1 or num_blocks <= 1 or n < _APPROX_TOPK_MIN_REDUCTION_SIZE
    ):
        vals, idx = jax.lax.top_k(work, k)
        return finalize(vals, idx)

    # Strided block reduction: split the last axis into `num_blocks` blocks of
    # `reduced_size`; the per-position max over blocks yields `reduced_size`
    # candidates. block b, position j  <-  flat index b*reduced_size + j.
    padded = num_blocks * reduced_size
    if padded != n:
        pad = jnp.broadcast_to(
            jnp.array(-jnp.inf, work.dtype), (*work.shape[:-1], padded - n)
        )
        work = jnp.concatenate([work, pad], axis=-1)
    blocks = work.reshape(*work.shape[:-1], num_blocks, reduced_size)
    cand_vals = jnp.max(blocks, axis=-2)
    best_block = jnp.argmax(blocks, axis=-2).astype(jnp.int32)
    cand_idx = best_block * reduced_size + jnp.arange(reduced_size, dtype=jnp.int32)

    # Aggregate the candidates: exact top-k(k), or (aggregate_to_topk=False) the
    # full candidate set sorted descending — top_k over all reduced_size entries.
    out_k = k if aggregate_to_topk else reduced_size
    top_vals, local = jax.lax.top_k(cand_vals, out_k)
    out_idx = jnp.take_along_axis(cand_idx, local, axis=-1)
    return finalize(top_vals, out_idx)


# ---------------------------------------------------------------------------
# Weight-only quantization (mps.quantize / dequantize / quantized_matmul)
# ---------------------------------------------------------------------------
# Affine group-wise quantization matching MLX's scheme (mlx/backend/cpu/
# quantized.cpp): each group of `group_size` values along the last axis is
# packed into uint32 words (group_size*bits/32 words per group) with a per-group
# fp `scale` and `bias` such that w ~= scale * q + bias. int4/int8 weight-only
# quant cuts the per-token weight bandwidth that bounds LLM decode. See #189.
#
# These are opt-in ops (no JAX primitive to intercept): on MPS they lower to
# MLX's fused quantized kernels; on other platforms a pure-JAX fallback runs the
# same affine math + bit packing so the same code is portable.


def _quant_pack_factor(bits):
    """Elements packed per uint32 word (power-of-2 bits, matches MLX)."""
    return 32 // bits


def _affine_quantize_math(w, group_size, bits):
    """Per-group affine quant. Returns (codes, scale, bias) matching MLX.

    `codes` has w's shape (integer values in [0, 2**bits-1], as float32);
    `scale`/`bias` have one entry per group along the last axis.
    """
    n_bins = float((1 << bits) - 1)
    eps = 1e-7
    shape = w.shape
    g = w.astype(jnp.float32).reshape(
        shape[:-1] + (shape[-1] // group_size, group_size)
    )
    w_min = jnp.min(g, axis=-1)
    w_max = jnp.max(g, axis=-1)
    mask = jnp.abs(w_min) > jnp.abs(w_max)
    scale = jnp.maximum((w_max - w_min) / n_bins, eps)
    scale = jnp.where(mask, scale, -scale)
    edge = jnp.where(mask, w_min, w_max)
    q0 = jnp.rint(edge / scale)
    nonzero = q0 != 0
    scale = jnp.where(nonzero, edge / jnp.where(nonzero, q0, 1.0), scale)
    bias = jnp.where(nonzero, edge, 0.0)
    codes = jnp.clip(jnp.rint((g - bias[..., None]) / scale[..., None]), 0.0, n_bins)
    return codes.reshape(shape), scale, bias


def _pack_uint32(codes, bits):
    """Pack integer `codes` (last axis) into uint32 words, k-th value at k*bits."""
    el = _quant_pack_factor(bits)
    grouped = codes.astype(jnp.uint32).reshape(
        codes.shape[:-1] + (codes.shape[-1] // el, el)
    )
    shifts = jnp.arange(el, dtype=jnp.uint32) * jnp.uint32(bits)
    return jnp.sum(grouped << shifts, axis=-1, dtype=jnp.uint32)


def _unpack_uint32(packed, bits):
    """Inverse of _pack_uint32: uint32 words -> integer codes (float32)."""
    el = _quant_pack_factor(bits)
    shifts = jnp.arange(el, dtype=jnp.uint32) * jnp.uint32(bits)
    bit_mask = jnp.uint32((1 << bits) - 1)
    codes = (packed[..., None] >> shifts) & bit_mask
    return codes.reshape(packed.shape[:-1] + (packed.shape[-1] * el,)).astype(
        jnp.float32
    )


# --- mps.quantize -----------------------------------------------------------
_quantize_p = core.Primitive("mps.quantize")
_quantize_p.multiple_results = True


def _quantize_abstract(w, *, group_size, bits):
    el = _quant_pack_factor(bits)
    d = w.shape[-1]
    packed_shape = w.shape[:-1] + (d // el,)
    group_shape = w.shape[:-1] + (d // group_size,)
    return (
        core.ShapedArray(packed_shape, jnp.uint32),
        core.ShapedArray(group_shape, w.dtype),
        core.ShapedArray(group_shape, w.dtype),
    )


_quantize_p.def_abstract_eval(_quantize_abstract)


def _quantize_impl(w, *, group_size, bits):
    codes, scale, bias = _affine_quantize_math(w, group_size, bits)
    return _pack_uint32(codes, bits), scale.astype(w.dtype), bias.astype(w.dtype)


_quantize_p.def_impl(_quantize_impl)


def _quantize_lowering(ctx, w, *, group_size, bits):
    return mlir.custom_call(
        call_target_name="mps.quantize",
        result_types=[_aval_to_ir_type(a) for a in ctx.avals_out],
        operands=[w],
        backend_config=f'{{"group_size": {group_size}, "bits": {bits}}}',
    ).results


# --- mps.dequantize ---------------------------------------------------------
_dequantize_p = core.Primitive("mps.dequantize")


def _dequantize_abstract(packed, scales, biases, *, group_size, bits):
    d = packed.shape[-1] * _quant_pack_factor(bits)
    return core.ShapedArray(packed.shape[:-1] + (d,), scales.dtype)


_dequantize_p.def_abstract_eval(_dequantize_abstract)


def _dequantize_impl(packed, scales, biases, *, group_size, bits):
    codes = _unpack_uint32(packed, bits)
    s = jnp.repeat(scales.astype(jnp.float32), group_size, axis=-1)
    b = jnp.repeat(biases.astype(jnp.float32), group_size, axis=-1)
    return (codes * s + b).astype(scales.dtype)


_dequantize_p.def_impl(_dequantize_impl)


def _dequantize_lowering(ctx, packed, scales, biases, *, group_size, bits):
    return mlir.custom_call(
        call_target_name="mps.dequantize",
        result_types=[_aval_to_ir_type(ctx.avals_out[0])],
        operands=[packed, scales, biases],
        backend_config=f'{{"group_size": {group_size}, "bits": {bits}}}',
    ).results


# --- mps.quantized_matmul ---------------------------------------------------
_quantized_matmul_p = core.Primitive("mps.quantized_matmul")


def _quantized_matmul_abstract(
    x, packed, scales, biases, *, transpose, group_size, bits
):
    if transpose:
        out = packed.shape[-2]  # w is [out, in], quantized along in
    else:
        out = packed.shape[-1] * _quant_pack_factor(bits)  # w is [in, out]
    return core.ShapedArray(x.shape[:-1] + (out,), x.dtype)


_quantized_matmul_p.def_abstract_eval(_quantized_matmul_abstract)


def _quantized_matmul_impl(x, packed, scales, biases, *, transpose, group_size, bits):
    w = _dequantize_impl(
        packed, scales, biases, group_size=group_size, bits=bits
    ).astype(x.dtype)
    if transpose:
        return jnp.matmul(x, jnp.swapaxes(w, -1, -2))
    return jnp.matmul(x, w)


_quantized_matmul_p.def_impl(_quantized_matmul_impl)


def _quantized_matmul_lowering(
    ctx, x, packed, scales, biases, *, transpose, group_size, bits
):
    return mlir.custom_call(
        call_target_name="mps.quantized_matmul",
        result_types=[_aval_to_ir_type(ctx.avals_out[0])],
        operands=[x, packed, scales, biases],
        backend_config=(
            f'{{"transpose": {1 if transpose else 0}, '
            f'"group_size": {group_size}, "bits": {bits}}}'
        ),
    ).results


# Bit widths whose packing (32 // bits elements per uint32 word) our packer and
# the MLX kernels support. Non-power-of-2 widths use a different MLX layout.
_SUPPORTED_QUANT_BITS = (4, 8)


def _check_quant_config(bits, group_size):
    """Validate quant params early so bad configs fail clearly, not deep in MLX."""
    if bits not in _SUPPORTED_QUANT_BITS:
        raise ValueError(
            f"quantization bits must be one of {_SUPPORTED_QUANT_BITS}, got {bits}"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size}")
    if (group_size * bits) % 32 != 0:
        raise ValueError(
            "group_size * bits must be a multiple of 32 for uint32 packing; got "
            f"group_size={group_size}, bits={bits}"
        )


def _check_group_shapes(name, packed, quant_dim, group_size, scales, biases):
    """Check per-group scales/biases match the quantized axis and packed prefix."""
    if quant_dim % group_size != 0:
        raise ValueError(
            f"{name}: quantized dim ({quant_dim}) must be a multiple of "
            f"group_size ({group_size})"
        )
    n_groups = quant_dim // group_size
    if scales.shape[-1] != n_groups or biases.shape[-1] != n_groups:
        raise ValueError(
            f"{name}: scales/biases last axis must be {n_groups} groups "
            f"(quantized dim {quant_dim} / group_size {group_size}); got "
            f"scales {scales.shape[-1]}, biases {biases.shape[-1]}"
        )
    prefix = tuple(packed.shape[:-1])
    if tuple(scales.shape[:-1]) != prefix or tuple(biases.shape[:-1]) != prefix:
        raise ValueError(
            f"{name}: scales/biases leading dims must match packed {prefix}; got "
            f"scales {tuple(scales.shape[:-1])}, biases {tuple(biases.shape[:-1])}"
        )


def quantize(w, *, group_size=64, bits=4):
    """Affine group-wise quantize `w` along its last axis (MLX layout).

    Returns ``(packed_uint32, scales, biases)`` such that
    ``dequantize(packed, scales, biases) ~= w``. ``w``'s last dim must be a
    multiple of ``group_size`` and ``group_size*bits`` a multiple of 32.
    """
    _check_quant_config(bits, group_size)
    if not jnp.issubdtype(w.dtype, jnp.floating):
        raise ValueError(
            f"quantize expects a floating-point weight, got dtype {w.dtype} "
            "(scales/biases would be truncated to that dtype)"
        )
    if w.shape[-1] % group_size != 0:
        raise ValueError(
            f"quantize: last axis ({w.shape[-1]}) must be a multiple of "
            f"group_size ({group_size})"
        )
    return _quantize_p.bind(w, group_size=group_size, bits=bits)


def dequantize(packed, scales, biases, *, group_size=64, bits=4):
    """Reconstruct a float array from a ``quantize`` result."""
    _check_quant_config(bits, group_size)
    quant_dim = packed.shape[-1] * _quant_pack_factor(bits)
    _check_group_shapes("dequantize", packed, quant_dim, group_size, scales, biases)
    return _dequantize_p.bind(packed, scales, biases, group_size=group_size, bits=bits)


def quantized_matmul(
    x, packed, scales, biases, *, transpose=True, group_size=64, bits=4
):
    """Multiply ``x`` by a quantized weight matrix using MLX's fused kernel.

    With ``transpose=True`` (default) the packed weight is ``[out, in]`` and the
    result is ``x @ wᵀ`` of shape ``x.shape[:-1] + (out,)``.
    """
    _check_quant_config(bits, group_size)
    # The grouped (quantized) axis is `in` when transpose=True and `out` otherwise.
    quant_dim = packed.shape[-1] * _quant_pack_factor(bits)
    contract = quant_dim if transpose else packed.shape[-2]
    if x.shape[-1] != contract:
        raise ValueError(
            f"quantized_matmul: x last axis ({x.shape[-1]}) must match the "
            f"contraction dim ({contract}) for transpose={transpose}"
        )
    _check_group_shapes(
        "quantized_matmul", packed, quant_dim, group_size, scales, biases
    )
    return _quantized_matmul_p.bind(
        x, packed, scales, biases, transpose=transpose, group_size=group_size, bits=bits
    )


def register_fused_ops():
    """Register MPS MLIR lowerings for fused custom_calls and related ops.

    Covers fused forward/backward primitives (sdpa, rms_norm, layer_norm, rope,
    gelu), linalg custom lowerings (eigh, qr, svd), CPU/GPU fallback lowerings
    for the fused primitives, and MPS-specific lowerings of JAX unary
    primitives (sinh, cosh, asin, acos, atan, asinh, acosh, atanh, erf,
    erf_inv) that would otherwise decompose through CHLO.
    """
    mlir.register_lowering(
        _metal_kernel_jit_p, _metal_kernel_jit_lowering, platform="mps"
    )
    mlir.register_lowering(
        _metal_kernel_lib_p, _metal_kernel_lib_lowering, platform="mps"
    )
    mlir.register_lowering(_sdpa_p, _sdpa_lowering, platform="mps")
    mlir.register_lowering(_sdpa_causal_p, _sdpa_causal_lowering, platform="mps")
    mlir.register_lowering(_rms_norm_p, _rms_norm_lowering, platform="mps")
    mlir.register_lowering(_layer_norm_p, _layer_norm_lowering, platform="mps")
    mlir.register_lowering(_rope_p, _rope_lowering, platform="mps")
    mlir.register_lowering(_gelu_p, _gelu_lowering, platform="mps")
    mlir.register_lowering(_quantize_p, _quantize_lowering, platform="mps")
    mlir.register_lowering(_dequantize_p, _dequantize_lowering, platform="mps")
    mlir.register_lowering(
        _quantized_matmul_p, _quantized_matmul_lowering, platform="mps"
    )

    # Backward lowerings for MPS.
    mlir.register_lowering(_sdpa_bwd_p, _sdpa_bwd_lowering, platform="mps")
    mlir.register_lowering(
        _sdpa_causal_bwd_p, _sdpa_causal_bwd_lowering, platform="mps"
    )
    mlir.register_lowering(_rms_norm_bwd_p, _rms_norm_bwd_lowering, platform="mps")
    mlir.register_lowering(_layer_norm_bwd_p, _layer_norm_bwd_lowering, platform="mps")
    mlir.register_lowering(_rope_bwd_p, _rope_bwd_lowering, platform="mps")
    mlir.register_lowering(_gelu_bwd_p, _gelu_bwd_lowering, platform="mps")

    # Linalg lowerings (eigh, qr, svd) for MPS platform.
    mlir.register_lowering(lax_linalg.eigh_p, _eigh_lowering, platform="mps")
    mlir.register_lowering(lax_linalg.qr_p, _qr_lowering, platform="mps")
    mlir.register_lowering(lax_linalg.svd_p, _svd_lowering, platform="mps")

    # Approximate top-k: a real MPS lowering (recall-driven strided reduction
    # with exact fallback) instead of JAX's exact non-TPU fallback. CPU/GPU keep
    # JAX's default lowering, so no fallback registration is needed here.
    mlir.register_lowering(
        lax_ann.approx_top_k_p,
        mlir.lower_fun(_approx_top_k_mps_impl, multiple_results=True),
        platform="mps",
    )

    # Intercept unary ops that JAX would normally lower to chlo.* primitives
    # (then decomposed to stablehlo.exp / add / etc. by CHLO legalization) and
    # route them directly to our native MLX implementation via a
    # stablehlo.custom_call @mhlo.* — which the C++ dispatcher handles in
    # HandleCustomCall with a single call to mlx::core::sinh/cosh/arcsin/....
    # Ops that JAX already lowers to a native stablehlo primitive we handle
    # (tan, atan2) are deliberately NOT in this list.
    _mps_native_unary_ops = [
        (lax_lax.sinh_p, "mhlo.sinh"),
        (lax_lax.cosh_p, "mhlo.cosh"),
        (lax_lax.asin_p, "mhlo.asin"),
        (lax_lax.acos_p, "mhlo.acos"),
        (lax_lax.atan_p, "mhlo.atan"),
        (lax_lax.asinh_p, "mhlo.asinh"),
        (lax_lax.acosh_p, "mhlo.acosh"),
        (lax_lax.atanh_p, "mhlo.atanh"),
        (lax_special.erf_p, "mhlo.erf"),
        (lax_special.erf_inv_p, "mhlo.erf_inv"),
    ]
    for prim, target in _mps_native_unary_ops:
        mlir.register_lowering(
            prim, _make_native_unary_lowering(target), platform="mps"
        )

    # logistic_p decomposes to 1/(1+exp(-x)) upstream; on MPS keep it as a
    # single stablehlo.logistic that dispatches to mlx::core::sigmoid.
    mlir.register_lowering(lax_lax.logistic_p, _logistic_lowering, platform="mps")

    # Fallback lowerings for non-MPS platforms (CPU, GPU).
    mlir.register_lowering(
        _sdpa_p,
        mlir.lower_fun(
            lambda q, k, v, mask, scale=1.0: _sdpa_impl(q, k, v, mask, scale=scale),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        _sdpa_causal_p,
        mlir.lower_fun(
            lambda q, k, v, scale=1.0: _sdpa_causal_impl(q, k, v, scale=scale),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        _rms_norm_p,
        mlir.lower_fun(
            lambda x, w, eps=1e-6: _rms_norm_impl(x, w, eps=eps), multiple_results=False
        ),
    )
    mlir.register_lowering(
        _layer_norm_p,
        mlir.lower_fun(
            lambda x, w, b, eps=1e-5: _layer_norm_impl(x, w, b, eps=eps),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        _rope_p,
        mlir.lower_fun(
            lambda x, offset, dims=0, traditional=False, base=10000.0, rope_scale=1.0: (
                _rope_impl(
                    x,
                    offset,
                    dims=dims,
                    traditional=traditional,
                    base=base,
                    rope_scale=rope_scale,
                )
            ),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        _gelu_p,
        mlir.lower_fun(lambda x: _gelu_impl_dispatch(x), multiple_results=False),
    )
    mlir.register_lowering(
        _quantize_p,
        mlir.lower_fun(
            lambda w, group_size=64, bits=4: _quantize_impl(
                w, group_size=group_size, bits=bits
            ),
            multiple_results=True,
        ),
    )
    mlir.register_lowering(
        _dequantize_p,
        mlir.lower_fun(
            lambda packed, scales, biases, group_size=64, bits=4: _dequantize_impl(
                packed, scales, biases, group_size=group_size, bits=bits
            ),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        _quantized_matmul_p,
        mlir.lower_fun(
            lambda x, packed, scales, biases, transpose=True, group_size=64, bits=4: (
                _quantized_matmul_impl(
                    x,
                    packed,
                    scales,
                    biases,
                    transpose=transpose,
                    group_size=group_size,
                    bits=bits,
                )
            ),
            multiple_results=False,
        ),
    )

    # Backward fallback lowerings for non-MPS platforms.
    mlir.register_lowering(
        _sdpa_bwd_p,
        mlir.lower_fun(
            lambda q, k, v, mask, g, scale=1.0: jax.vjp(
                lambda q, k, v: _sdpa_impl(q, k, v, mask, scale=scale), q, k, v
            )[1](g),
            multiple_results=True,
        ),
    )
    mlir.register_lowering(
        _sdpa_causal_bwd_p,
        mlir.lower_fun(
            lambda q, k, v, g, scale=1.0: jax.vjp(
                lambda q, k, v: _sdpa_causal_impl(q, k, v, scale=scale), q, k, v
            )[1](g),
            multiple_results=True,
        ),
    )
    mlir.register_lowering(
        _rms_norm_bwd_p,
        mlir.lower_fun(
            lambda x, w, g, eps=1e-6: jax.vjp(
                lambda x, w: _rms_norm_impl(x, w, eps=eps), x, w
            )[1](g),
            multiple_results=True,
        ),
    )
    mlir.register_lowering(
        _layer_norm_bwd_p,
        mlir.lower_fun(
            lambda x, w, b, g, eps=1e-5: jax.vjp(
                lambda x, w, b: _layer_norm_impl(x, w, b, eps=eps), x, w, b
            )[1](g),
            multiple_results=True,
        ),
    )
    mlir.register_lowering(
        _rope_bwd_p,
        mlir.lower_fun(
            lambda x, offset, g, dims=0, traditional=False, base=10000.0, rope_scale=1.0: (
                jax.vjp(
                    lambda x: _rope_impl(
                        x,
                        offset,
                        dims=dims,
                        traditional=traditional,
                        base=base,
                        rope_scale=rope_scale,
                    ),
                    x,
                )[1](g)[0]
            ),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        _gelu_bwd_p,
        mlir.lower_fun(
            lambda x, g: jax.vjp(lambda x: _gelu_impl_dispatch(x), x)[1](g)[0],
            multiple_results=False,
        ),
    )


# Store originals before any patching.
_gelu_original = None
_sdpa_original = None


class PatchConflictError(RuntimeError):
    """Raised when another library has already monkey-patched a function we need."""

    pass


def patch_jax_functions():
    """Monkey-patch jax.nn.gelu and jax.nn.dot_product_attention.

    On MPS, routes through fused MLX kernels via custom_call primitives.
    On other platforms, the fallback lowering decomposes to standard JAX ops,
    so behavior is identical to unpatched JAX.

    Raises PatchConflictError if the functions have already been patched by
    another library.
    """
    import jax.nn as jnn
    from jax._src.nn import functions as nn_functions

    global _gelu_original, _sdpa_original

    # --- GELU ---
    original_gelu = nn_functions.gelu
    if not getattr(original_gelu, "_mps_patched", False):
        if getattr(original_gelu, "_patched", False) or not hasattr(
            original_gelu, "__module__"
        ):
            raise PatchConflictError(
                "jax.nn.gelu has already been monkey-patched by another library. "
                "The MPS plugin cannot safely patch it. Disable the other patch or "
                "use jax_plugins.mps.ops.gelu() directly."
            )
        _gelu_original = original_gelu

        def _patched_gelu(x, approximate=True):
            if approximate:
                return gelu(x)
            return _gelu_original(x, approximate=False)

        _patched_gelu._mps_patched = True
        _patched_gelu.__doc__ = original_gelu.__doc__
        nn_functions.gelu = _patched_gelu
        jnn.gelu = _patched_gelu

    # --- dot_product_attention ---
    original_sdpa = nn_functions.dot_product_attention
    if not getattr(original_sdpa, "_mps_patched", False):
        if getattr(original_sdpa, "_patched", False) or not hasattr(
            original_sdpa, "__module__"
        ):
            raise PatchConflictError(
                "jax.nn.dot_product_attention has already been monkey-patched by "
                "another library. The MPS plugin cannot safely patch it. Disable "
                "the other patch or use jax_plugins.mps.ops.sdpa() directly."
            )
        _sdpa_original = original_sdpa

        def _patched_dot_product_attention(
            query,
            key,
            value,
            bias=None,
            mask=None,
            *,
            scale=None,
            is_causal=False,
            query_seq_lengths=None,
            key_value_seq_lengths=None,
            local_window_size=None,
            implementation=None,
            return_residual=False,
        ):
            # Fall back for features we can't fuse: bias, seq lengths, etc.
            # We CAN handle boolean masks via the fused masked SDPA path.
            if (
                bias is not None
                or query_seq_lengths is not None
                or key_value_seq_lengths is not None
                or local_window_size is not None
                or return_residual
                or implementation is not None
            ):
                return _sdpa_original(
                    query,
                    key,
                    value,
                    bias,
                    mask,
                    scale=scale,
                    is_causal=is_causal,
                    query_seq_lengths=query_seq_lengths,
                    key_value_seq_lengths=key_value_seq_lengths,
                    local_window_size=local_window_size,
                    implementation=implementation,
                    return_residual=return_residual,
                )

            # Normalize to 4D: (B, T, N, H).
            q = jnp.asarray(query)
            k = jnp.asarray(key)
            v = jnp.asarray(value)

            # Only handle 3D/4D inputs; fall back for other ranks.
            if q.ndim not in (3, 4):
                return _sdpa_original(
                    query,
                    key,
                    value,
                    mask=mask,
                    scale=scale,
                    is_causal=is_causal,
                )

            squeeze = q.ndim == 3
            if squeeze:
                q = q[None]
                k = k[None]
                v = v[None]

            B, T, N, H = q.shape
            _, S, K, _ = k.shape

            # Fall back if GQA head counts aren't divisible.
            if K < N and N % K != 0:
                return _sdpa_original(
                    query,
                    key,
                    value,
                    mask=mask,
                    scale=scale,
                    is_causal=is_causal,
                )

            if scale is None:
                scale = H**-0.5

            # Transpose to (B, N, T, H) for our SDPA primitive.
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            # Expand KV heads for GQA.
            if K < N:
                k = jnp.repeat(k, N // K, axis=1)
                v = jnp.repeat(v, N // K, axis=1)

            # Prepare mask for fused SDPA: broadcast to (B, N, T, S).
            fused_mask = None
            if mask is not None:
                m = jnp.asarray(mask)
                if squeeze:
                    m = m[None]
                # Input mask is (B, 1, 1, S) or (B, T, N, S) in BTNH layout.
                # Transpose to (B, N, T, S) for our SDPA primitive.
                if m.ndim == 4:
                    m = m.transpose(0, 2, 1, 3)
                # Broadcast to full shape so MLX gets a concrete mask array.
                fused_mask = jnp.broadcast_to(m, (B, N, T, S))

            out = sdpa(
                q, k, v, scale=float(scale), is_causal=is_causal, mask=fused_mask
            )

            # Back to (B, T, N, H).
            out = out.transpose(0, 2, 1, 3)
            if squeeze:
                out = out[0]
            return out

        _patched_dot_product_attention._mps_patched = True
        _patched_dot_product_attention.__doc__ = original_sdpa.__doc__
        nn_functions.dot_product_attention = _patched_dot_product_attention
        jnn.dot_product_attention = _patched_dot_product_attention

    # --- Flax LayerNorm ---
    try:
        from flax import nnx as _nnx

        _original_ln_call = _nnx.LayerNorm.__call__
        if not getattr(_original_ln_call, "_mps_patched", False):

            def _patched_layer_norm_call(self, x, *, mask=None):
                ra = self.reduction_axes
                fa = self.feature_axes
                if (
                    mask is None
                    and self.use_bias
                    and (ra == -1 or ra == (-1,))
                    and (fa == -1 or fa == (-1,))
                ):
                    return layer_norm(
                        x, self.scale.value, self.bias.value, eps=self.epsilon
                    )
                return _original_ln_call(self, x, mask=mask)

            _patched_layer_norm_call._mps_patched = True
            _nnx.LayerNorm.__call__ = _patched_layer_norm_call  # type: ignore[assignment]
    except ImportError:
        pass
