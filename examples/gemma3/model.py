"""Gemma 3 text transformer architecture in Flax NNX."""

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from jax_plugins.mps.ops import rms_norm, rope


@dataclasses.dataclass
class Gemma3Config:
    vocab_size: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    query_pre_attn_scalar: float = 256.0
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    dtype: jnp.dtype = jnp.float32


KVCache = list[tuple[jax.Array, jax.Array]]


class RMSNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nnx.Param(jnp.zeros(dim))
        self.eps = eps

    def __call__(self, x):
        w = self.weight[...]
        return rms_norm(x, jnp.ones_like(w) + w, eps=self.eps)


class Gemma3Attention(nnx.Module):
    def __init__(self, config, layer_idx, *, rngs):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.query_pre_attn_scalar**-0.5
        self.layer_idx = layer_idx
        self.is_sliding = (layer_idx + 1) % config.sliding_window_pattern != 0
        self.sliding_window = config.sliding_window

        # Different RoPE base for sliding vs global layers.
        if self.is_sliding:
            self.rope_theta = config.rope_local_base_freq
        else:
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

        # QK norms (new in Gemma 3).
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

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

        # QK norms before RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)

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
                # Mask: attend only to valid positions.
                S = k.shape[2]
                valid = jnp.arange(S)[None, :] < (cache_index + T)
                if self.is_sliding:
                    # Sliding window: also mask positions too far back.
                    in_window = jnp.arange(S)[None, :] >= (
                        cache_index + T - self.sliding_window
                    )
                    valid = valid & in_window
                mask = valid[:, None, None, :]  # (B, 1, 1, S)
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
            out = jax.nn.dot_product_attention(q, k, v, mask=mask, scale=self.scale)
        elif kv_cache is None:
            out = jax.nn.dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale
            )
        else:
            out = jax.nn.dot_product_attention(q, k, v, scale=self.scale)
        return self.o_proj(out.reshape(B, T, -1)), new_kv


class Gemma3MLP(nnx.Module):
    def __init__(self, config, *, rngs):
        dt = config.dtype
        # Fused gate+up projection.
        self.gate_up_proj = nnx.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            use_bias=False,
            dtype=dt,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dt,
            rngs=rngs,
        )

    def __call__(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return self.down_proj(jax.nn.gelu(gate, approximate=True) * up)


def _clip_residual(x, y):
    """Add residual with float16 overflow protection.

    Gemma 3's large norm weights cause hidden states to exceed sqrt(float16_max),
    which makes x^2 overflow in RMSNorm. Clipping the residual sum prevents this.
    """
    if x.dtype != jnp.float16:
        return x + y
    bound = jnp.finfo(jnp.float16).max
    return jnp.clip(
        x.astype(jnp.float32) + y.astype(jnp.float32), -bound, bound
    ).astype(jnp.float16)


class Gemma3DecoderLayer(nnx.Module):
    def __init__(self, config, layer_idx, *, rngs):
        self.self_attn = Gemma3Attention(config, layer_idx, rngs=rngs)
        self.mlp = Gemma3MLP(config, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(self, x, pos_offset, kv_cache=None, cache_index=None):
        attn_out, new_kv = self.self_attn(
            self.input_layernorm(x), pos_offset, kv_cache, cache_index
        )
        x = _clip_residual(x, self.post_attention_layernorm(attn_out))
        x = _clip_residual(
            x,
            self.post_feedforward_layernorm(
                self.mlp(self.pre_feedforward_layernorm(x))
            ),
        )
        return x, new_kv


class Gemma3(nnx.Module):
    def __init__(self, config, *, rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(
            config.vocab_size, config.hidden_size, dtype=config.dtype, rngs=rngs
        )
        self.layers = nnx.List(
            Gemma3DecoderLayer(config, layer_idx=i, rngs=rngs)
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

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
        # Tied embeddings — reuse embedding matrix as logit projection.
        return x @ self.embed_tokens.embedding[...].T, new_kv_cache
