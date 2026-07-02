"""Gemma transformer architecture in Flax NNX."""

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from jax_plugins.mps.ops import quantize, quantized_matmul, rms_norm, rope


@dataclasses.dataclass
class GemmaConfig:
    vocab_size: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    # Optionally quantize the (tied) LM-head matmul to int8/int4. The LM head is
    # ~21% of the per-token weight read in gemma-2b; the fp embedding is still
    # kept for the cheap token-lookup gather.
    quantize_lm_head: bool = False
    lm_head_bits: int = 8
    quant_group_size: int = 64
    # Optionally quantize the MLP dense layers (gate_up + down = ~72% of the
    # per-token weight read). 0 = off; otherwise the bit width (4 or 8) for the
    # gate_up projection. down_proj is more sensitive, so it uses
    # quantize_down_bits instead (mixed recipe: gate_up int4, down_proj int8).
    quantize_mlp_bits: int = 0
    quantize_down_bits: int = 8


KVCache = list[tuple[jax.Array, jax.Array]]


class RMSNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nnx.Param(jnp.zeros(dim))
        self.eps = eps

    def __call__(self, x):
        w = self.weight[...]
        return rms_norm(x, jnp.ones_like(w) + w, eps=self.eps)


class QuantizedLinear(nnx.Module):
    """Bias-free Linear whose weight is int4/int8 weight-only quantized.

    Holds MLX-packed weights (uint32) + per-group fp scale/bias and runs the
    fused mps.quantized_matmul. The packed weight is ``[out, in]`` (quantized
    along ``in``); load it with ``set_quantized`` from an ``[out, in]`` matrix.
    """

    def __init__(self, in_features, out_features, *, bits, group_size, dtype):
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.packed = nnx.Param(
            jnp.zeros((out_features, in_features * bits // 32), jnp.uint32)
        )
        self.scales = nnx.Param(
            jnp.zeros((out_features, in_features // group_size), dtype)
        )
        self.biases = nnx.Param(
            jnp.zeros((out_features, in_features // group_size), dtype)
        )

    def set_quantized(self, weight):
        """Quantize an ``[out, in]`` weight and store the packed params."""
        packed, scales, biases = quantize(
            weight, group_size=self.group_size, bits=self.bits
        )
        self.packed.set_value(packed)
        self.scales.set_value(scales)
        self.biases.set_value(biases)

    def __call__(self, x):
        *lead, k = x.shape
        y = quantized_matmul(
            x.reshape(-1, k),
            self.packed[...],
            self.scales[...],
            self.biases[...],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        return y.reshape(*lead, self.out_features)


def _dense(in_features, out_features, config, *, rngs, bits=None):
    """A gemma dense layer: QuantizedLinear if `bits`, else fp nnx.Linear.

    `bits` defaults to config.quantize_mlp_bits; pass an override (e.g. for the
    higher-precision down_proj in the mixed recipe).
    """
    if bits is None:
        bits = config.quantize_mlp_bits
    if bits:
        return QuantizedLinear(
            in_features,
            out_features,
            bits=bits,
            group_size=config.quant_group_size,
            dtype=config.dtype,
        )
    return nnx.Linear(
        in_features, out_features, use_bias=False, dtype=config.dtype, rngs=rngs
    )


class GemmaAttention(nnx.Module):
    def __init__(self, config, *, rngs):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        dt = config.dtype
        self.q_proj = nnx.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=dt,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            dtype=dt,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            dtype=dt,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dt,
            rngs=rngs,
        )

    def __call__(self, x, pos_offset, kv_cache=None, cache_index=None):
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, T, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        q = rope(q, dims=self.head_dim, base=self.rope_theta, offset=pos_offset)
        k = rope(k, dims=self.head_dim, base=self.rope_theta, offset=pos_offset)

        mask = None
        if kv_cache is not None:
            if cache_index is not None:
                # Static cache: write new KV at cache_index, use full buffer.
                k_cache = jax.lax.dynamic_update_slice(
                    kv_cache[0], k, (0, 0, cache_index, 0)
                )
                v_cache = jax.lax.dynamic_update_slice(
                    kv_cache[1], v, (0, 0, cache_index, 0)
                )
                k = k_cache
                v = v_cache
                new_kv = (k_cache, v_cache)
                # Mask: attend only to positions < cache_index + T.
                S = k.shape[2]
                mask = jnp.arange(S)[None, :] < (cache_index + T)
                mask = mask[:, None, None, :]  # (B, 1, 1, S)
            else:
                k = jnp.concatenate([kv_cache[0], k], axis=2)
                v = jnp.concatenate([kv_cache[1], v], axis=2)
                new_kv = (k, v)
        else:
            new_kv = (k, v)

        # Back to (B, T, N, H) for jax.nn.dot_product_attention.
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if mask is not None:
            out = jax.nn.dot_product_attention(q, k, v, mask=mask)
        elif kv_cache is None:
            out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        else:
            out = jax.nn.dot_product_attention(q, k, v)
        return self.o_proj(out.reshape(B, T, -1)), new_kv


class GemmaMLP(nnx.Module):
    def __init__(self, config, *, rngs):
        # Fused gate+up projection: one matmul instead of two.
        self.gate_up_proj = _dense(
            config.hidden_size, 2 * config.intermediate_size, config, rngs=rngs
        )
        # down_proj is more sensitive: keep it at quantize_down_bits (int8) when
        # the MLP is quantized (mixed recipe), else fp.
        down_bits = config.quantize_down_bits if config.quantize_mlp_bits else 0
        self.down_proj = _dense(
            config.intermediate_size,
            config.hidden_size,
            config,
            rngs=rngs,
            bits=down_bits,
        )

    def __call__(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return self.down_proj(jax.nn.gelu(gate, approximate=True) * up)


class GemmaDecoderLayer(nnx.Module):
    def __init__(self, config, *, rngs):
        self.self_attn = GemmaAttention(config, rngs=rngs)
        self.mlp = GemmaMLP(config, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x, pos_offset, kv_cache=None, cache_index=None):
        attn_out, new_kv = self.self_attn(
            self.input_layernorm(x), pos_offset, kv_cache, cache_index
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Gemma(nnx.Module):
    def __init__(self, config, *, rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(
            config.vocab_size, config.hidden_size, dtype=config.dtype, rngs=rngs
        )
        self.layers = nnx.List(
            GemmaDecoderLayer(config, rngs=rngs)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Quantized LM-head weights (packed uint32 + per-group scale/bias),
        # populated at load from the tied embedding. Placeholders so the module
        # structure is built by jax.eval_shape before real weights are set.
        if config.quantize_lm_head:
            gs, bits = config.quant_group_size, config.lm_head_bits
            v, h = config.vocab_size, config.hidden_size
            self.lm_head_packed = nnx.Param(jnp.zeros((v, h * bits // 32), jnp.uint32))
            self.lm_head_scales = nnx.Param(jnp.zeros((v, h // gs), config.dtype))
            self.lm_head_biases = nnx.Param(jnp.zeros((v, h // gs), config.dtype))

    def __call__(self, input_ids, kv_cache=None, pos_offset=0, cache_index=None):
        x = self.embed_tokens(input_ids)
        x = x * jnp.sqrt(jnp.array(self.config.hidden_size, dtype=x.dtype))
        new_kv_cache: KVCache = []
        for i, layer in enumerate(self.layers):
            x, new_kv = layer(
                x, pos_offset, kv_cache[i] if kv_cache else None, cache_index
            )
            new_kv_cache.append(new_kv)
        x = self.norm(x)
        if self.config.quantize_lm_head:
            b, t, h = x.shape
            logits = quantized_matmul(
                x.reshape(b * t, h),
                self.lm_head_packed[...],
                self.lm_head_scales[...],
                self.lm_head_biases[...],
                transpose=True,
                group_size=self.config.quant_group_size,
                bits=self.config.lm_head_bits,
            ).reshape(b, t, self.config.vocab_size)
        else:
            logits = x @ self.embed_tokens.embedding[...].T
        return logits, new_kv_cache
