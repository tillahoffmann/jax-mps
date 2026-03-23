"""Fused MLX operations exposed as JAX primitives via custom_call.

These primitives emit stablehlo.custom_call ops that the MPS backend
intercepts and dispatches to mlx::core::fast:: fused Metal kernels.

The forward pass uses fused kernels; the backward pass falls through to
JAX's standard autodiff on the decomposed implementation.
"""

# pyright: reportArgumentType=false, reportOptionalCall=false
# pyright: reportFunctionMemberAccess=false, reportCallIssue=false

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.interpreters import mlir

# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention (mps.sdpa)
# ---------------------------------------------------------------------------

_sdpa_p = core.Primitive("mps.sdpa")
_sdpa_p.multiple_results = False


def _sdpa_abstract(q, k, v, *, scale):
    return core.ShapedArray(q.shape, q.dtype)


_sdpa_p.def_abstract_eval(_sdpa_abstract)


def _sdpa_impl(q, k, v, *, scale):
    """Pure JAX fallback for non-MPS platforms."""
    attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.matmul(attn, v)


_sdpa_p.def_impl(_sdpa_impl)


def _sdpa_lowering(ctx, q, k, v, *, scale):
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
    return mlir.custom_call(
        call_target_name="mps.sdpa",
        result_types=[result_type],
        operands=[q, k, v],
        backend_config=f'{{"scale": {scale}}}',
    ).results


# Causal variant.
_sdpa_causal_p = core.Primitive("mps.sdpa_causal")
_sdpa_causal_p.multiple_results = False
_sdpa_causal_p.def_abstract_eval(_sdpa_abstract)


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


def sdpa(q, k, v, *, scale=None, is_causal=False):
    """Scaled dot-product attention using fused MLX kernel on MPS.

    Args:
        q: Queries, shape (B, N, T, H).
        k: Keys, shape (B, N_kv, S, H).
        v: Values, shape (B, N_kv, S, H).
        scale: Attention scale factor. Defaults to 1/sqrt(H).
        is_causal: Whether to apply causal (lower-triangular) masking.

    Returns:
        Output of shape (B, N, T, H).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scale = float(scale)
    if is_causal:
        return _sdpa_causal_with_grad(q, k, v, scale)
    return _sdpa_with_grad(q, k, v, scale)


def _sdpa_with_grad(q, k, v, scale):
    @jax.custom_vjp
    def fwd(q, k, v):
        return _sdpa_p.bind(q, k, v, scale=scale)

    def fwd_rule(q, k, v):
        return fwd(q, k, v), (q, k, v)

    def bwd_rule(res, g):
        q, k, v = res
        return jax.vjp(lambda q, k, v: _sdpa_impl(q, k, v, scale=scale), q, k, v)[1](g)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(q, k, v)


def _sdpa_causal_with_grad(q, k, v, scale):
    @jax.custom_vjp
    def fwd(q, k, v):
        return _sdpa_causal_p.bind(q, k, v, scale=scale)

    def fwd_rule(q, k, v):
        return fwd(q, k, v), (q, k, v)

    def bwd_rule(res, g):
        q, k, v = res
        return jax.vjp(
            lambda q, k, v: _sdpa_causal_impl(q, k, v, scale=scale), q, k, v
        )[1](g)

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


def rms_norm(x, weight, *, eps=1e-6):
    """RMS normalization using fused MLX kernel on MPS.

    Computes: x / sqrt(mean(x^2) + eps) * weight
    """
    eps = float(eps)

    @jax.custom_vjp
    def fwd(x, weight):
        return _rms_norm_p.bind(x, weight, eps=eps)

    def fwd_rule(x, weight):
        return fwd(x, weight), (x, weight)

    def bwd_rule(res, g):
        x, weight = res
        return jax.vjp(lambda x, w: _rms_norm_impl(x, w, eps=eps), x, weight)[1](g)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(x, weight)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (mps.rope)
# ---------------------------------------------------------------------------

_rope_p = core.Primitive("mps.rope")
_rope_p.multiple_results = False


def _rope_abstract(x, *, dims, traditional, base, rope_scale, offset):
    return core.ShapedArray(x.shape, x.dtype)


_rope_p.def_abstract_eval(_rope_abstract)


def _rope_impl(x, *, dims, traditional, base, rope_scale, offset):
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
    result = x[..., :dims] * cos + rotated * sin
    if dims < x.shape[-1]:
        result = jnp.concatenate([result, x[..., dims:]], axis=-1)
    return result


_rope_p.def_impl(_rope_impl)


def _rope_lowering(ctx, x, *, dims, traditional, base, rope_scale, offset):
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
    traditional_str = "true" if traditional else "false"
    return mlir.custom_call(
        call_target_name="mps.rope",
        result_types=[result_type],
        operands=[x],
        backend_config=(
            f'{{"dims": {dims}, "traditional": {traditional_str}, '
            f'"base": {base}, "rope_scale": {rope_scale}, "offset": {offset}}}'
        ),
    ).results


def rope(x, *, dims, base=10000.0, scale=1.0, offset=0, traditional=False):
    """Rotary position embeddings using fused MLX kernel on MPS."""
    dims = int(dims)
    traditional = bool(traditional)
    base = float(base)
    rope_scale = float(scale)
    offset = int(offset)

    @jax.custom_vjp
    def fwd(x):
        return _rope_p.bind(
            x,
            dims=dims,
            traditional=traditional,
            base=base,
            rope_scale=rope_scale,
            offset=offset,
        )

    def fwd_rule(x):
        return fwd(x), (x,)

    def bwd_rule(res, g):
        (x,) = res
        return jax.vjp(
            lambda x: _rope_impl(
                x,
                dims=dims,
                traditional=traditional,
                base=base,
                rope_scale=rope_scale,
                offset=offset,
            ),
            x,
        )[1](g)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(x)


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


def gelu(x):
    """Approximate GELU using fused MLX kernel on MPS."""

    @jax.custom_vjp
    def fwd(x):
        return _gelu_p.bind(x)

    def fwd_rule(x):
        return fwd(x), (x,)

    def bwd_rule(res, g):
        (x,) = res
        return jax.vjp(lambda x: _gelu_impl_dispatch(x), x)[1](g)

    fwd.defvjp(fwd_rule, bwd_rule)
    return fwd(x)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_fused_ops():
    """Register MLIR lowerings for all fused ops on the MPS platform."""
    mlir.register_lowering(_sdpa_p, _sdpa_lowering, platform="mps")
    mlir.register_lowering(_sdpa_causal_p, _sdpa_causal_lowering, platform="mps")
    mlir.register_lowering(_rms_norm_p, _rms_norm_lowering, platform="mps")
    mlir.register_lowering(_rope_p, _rope_lowering, platform="mps")
    mlir.register_lowering(_gelu_p, _gelu_lowering, platform="mps")

    # Fallback lowerings for non-MPS platforms (CPU, GPU).
    mlir.register_lowering(
        _sdpa_p,
        mlir.lower_fun(
            lambda q, k, v, scale=1.0: _sdpa_impl(q, k, v, scale=scale),
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
        _rope_p,
        mlir.lower_fun(
            lambda x, dims=0, traditional=False, base=10000.0, rope_scale=1.0, offset=0: (
                _rope_impl(
                    x,
                    dims=dims,
                    traditional=traditional,
                    base=base,
                    rope_scale=rope_scale,
                    offset=offset,
                )
            ),
            multiple_results=False,
        ),
    )
    mlir.register_lowering(
        _gelu_p,
        mlir.lower_fun(lambda x: _gelu_impl_dispatch(x), multiple_results=False),
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
            # Only intercept simple cases; fall back for masks, bias, implementation, etc.
            if (
                bias is not None
                or mask is not None
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

            out = sdpa(q, k, v, scale=float(scale), is_causal=is_causal)

            # Back to (B, T, N, H).
            out = out.transpose(0, 2, 1, 3)
            if squeeze:
                out = out[0]
            return out

        _patched_dot_product_attention._mps_patched = True
        _patched_dot_product_attention.__doc__ = original_sdpa.__doc__
        nn_functions.dot_product_attention = _patched_dot_product_attention
        jnn.dot_product_attention = _patched_dot_product_attention
