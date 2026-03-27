"""Fused MLX operations exposed as JAX primitives via custom_call.

These primitives emit stablehlo.custom_call ops that the MPS backend
intercepts and dispatches to mlx::core::fast:: fused Metal kernels.

On MPS, both forward and backward passes run as fused MLX kernels (the
backward uses mlx::core::vjp). On non-MPS platforms, fallback lowerings
decompose to standard JAX ops.
"""

# pyright: reportArgumentType=false, reportOptionalCall=false
# pyright: reportFunctionMemberAccess=false, reportCallIssue=false

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.interpreters import mlir
from jax._src.lax import linalg as lax_linalg

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
    """Pure JAX fallback for non-MPS platforms."""
    attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
    attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.matmul(attn, v)


_sdpa_p.def_impl(_sdpa_impl)


def _sdpa_lowering(ctx, q, k, v, mask, *, scale):
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
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
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
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
        mask: Optional boolean mask, shape broadcastable to (B, N, T, S).
            True means attend, False means ignore.

    Returns:
        Output of shape (B, N, T, H).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scale = float(scale)
    if is_causal:
        return _sdpa_causal_with_grad(q, k, v, scale)
    if mask is None:
        mask = jnp.bool_(True)
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
        result_types=[mlir.aval_to_ir_type(a) for a in avals],
        operands=[q, k, v, mask, g],
        backend_config=f'{{"scale": {scale}}}',
    ).results


def _sdpa_causal_bwd_lowering(ctx, q, k, v, g, *, scale):
    avals = ctx.avals_out
    return mlir.custom_call(
        call_target_name="mps.sdpa_causal_bwd",
        result_types=[mlir.aval_to_ir_type(a) for a in avals],
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
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
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
        result_types=[mlir.aval_to_ir_type(a) for a in avals],
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
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
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
        result_types=[mlir.aval_to_ir_type(a) for a in avals],
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
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
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
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
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
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
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
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
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
    v_type = mlir.aval_to_ir_type(v_aval)
    w_type = mlir.aval_to_ir_type(w_aval)
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
    q_type = mlir.aval_to_ir_type(q_aval)
    r_type = mlir.aval_to_ir_type(r_aval)
    return mlir.custom_call(
        call_target_name="mps.qr",
        result_types=[q_type, r_type],
        operands=[operand],
        backend_config=f'{{"full_matrices": {str(full_matrices).lower()}}}',
    ).results


def _svd_lowering(
    ctx, operand, *, full_matrices, compute_uv, subset_by_index, algorithm
):
    """MPS lowering for svd: emit a custom_call @mps.svd."""
    if subset_by_index is not None:
        raise NotImplementedError("subset_by_index not supported on MPS")
    if algorithm is not None:
        raise NotImplementedError("algorithm selection not supported on MPS")
    operand_aval = ctx.avals_in[0]
    m, n = operand_aval.shape[-2], operand_aval.shape[-1]
    if m < n:
        raise NotImplementedError(
            f"MPS SVD does not yet support wide matrices (M < N). "
            f"Got {m}×{n}. Transpose the input first."
        )
    fm = "true" if full_matrices else "false"
    if compute_uv:
        # JAX svd_p abstract eval returns (s, u, vt) – note s first!
        s_aval, u_aval, vt_aval = ctx.avals_out
        return mlir.custom_call(
            call_target_name="mps.svd",
            result_types=[
                mlir.aval_to_ir_type(s_aval),
                mlir.aval_to_ir_type(u_aval),
                mlir.aval_to_ir_type(vt_aval),
            ],
            operands=[operand],
            backend_config=f'{{"compute_uv": true, "full_matrices": {fm}}}',
        ).results
    else:
        (s_aval,) = ctx.avals_out
        return mlir.custom_call(
            call_target_name="mps.svd",
            result_types=[mlir.aval_to_ir_type(s_aval)],
            operands=[operand],
            backend_config=f'{{"compute_uv": false, "full_matrices": {fm}}}',
        ).results


def register_fused_ops():
    """Register MLIR lowerings for all fused ops on the MPS platform."""
    mlir.register_lowering(_sdpa_p, _sdpa_lowering, platform="mps")
    mlir.register_lowering(_sdpa_causal_p, _sdpa_causal_lowering, platform="mps")
    mlir.register_lowering(_rms_norm_p, _rms_norm_lowering, platform="mps")
    mlir.register_lowering(_layer_norm_p, _layer_norm_lowering, platform="mps")
    mlir.register_lowering(_rope_p, _rope_lowering, platform="mps")
    mlir.register_lowering(_gelu_p, _gelu_lowering, platform="mps")

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
