"""Gemma transformer architecture in Flax NNX."""

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from jax_plugins.mps.ops import rms_norm, rope


@dataclasses.dataclass
class GemmaConfig:
    vocab_size: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6


# KV cache is a list of (k, v) tuples, one per layer.
KVCache = list[tuple[jax.Array, jax.Array]]


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = nnx.Param(jnp.zeros(dim))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        # Gemma uses (1 + weight) scaling; pass the effective weight to the fused kernel.
        return rms_norm(x, 1 + self.weight[...], eps=self.eps)


class GemmaAttention(nnx.Module):
    def __init__(self, config: GemmaConfig, *, rngs: nnx.Rngs):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nnx.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            use_bias=False,
            rngs=rngs,
        )
        self.rope_theta = config.rope_theta

    def __call__(
        self,
        x: jax.Array,
        pos_offset: int,
        kv_cache: Optional[tuple[jax.Array, jax.Array]] = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        batch, seq_len, _ = x.shape
        # Project and reshape to (B, N, T, H) for RoPE and SDPA.
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)  # (B, N, T, H)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Fused RoPE: operates on (..., T, D) layout.
        q = rope(q, dims=self.head_dim, base=self.rope_theta, offset=pos_offset)
        k = rope(k, dims=self.head_dim, base=self.rope_theta, offset=pos_offset)

        # Append to KV cache (stored as (B, N, S, H)).
        if kv_cache is not None:
            k = jnp.concatenate([kv_cache[0], k], axis=2)
            v = jnp.concatenate([kv_cache[1], v], axis=2)
        new_kv = (k, v)

        # Transpose back to (B, T, N, H) for jax.nn.dot_product_attention.
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Use standard JAX SDPA — monkey-patched to use fused MPS kernel.
        out = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=(kv_cache is None),
        )
        out = out.reshape(batch, seq_len, -1)

        return self.o_proj(out), new_kv


class GemmaMLP(nnx.Module):
    def __init__(self, config: GemmaConfig, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(
            config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs
        )
        self.up_proj = nnx.Linear(
            config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size, config.hidden_size, use_bias=False, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(
            jax.nn.gelu(self.gate_proj(x), approximate=True) * self.up_proj(x)
        )


class GemmaDecoderLayer(nnx.Module):
    def __init__(self, config: GemmaConfig, *, rngs: nnx.Rngs):
        self.self_attn = GemmaAttention(config, rngs=rngs)
        self.mlp = GemmaMLP(config, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        x: jax.Array,
        pos_offset: int,
        kv_cache: Optional[tuple[jax.Array, jax.Array]] = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        attn_out, new_kv = self.self_attn(self.input_layernorm(x), pos_offset, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Gemma(nnx.Module):
    def __init__(self, config: GemmaConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embed_tokens = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.layers = nnx.List(
            GemmaDecoderLayer(config, rngs=rngs)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        input_ids: jax.Array,
        kv_cache: Optional[KVCache] = None,
        pos_offset: int = 0,
    ) -> tuple[jax.Array, KVCache]:
        batch, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        # Gemma scales embeddings by sqrt(hidden_size).
        x = x * jnp.sqrt(jnp.float32(self.config.hidden_size))

        new_kv_cache: KVCache = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x, new_kv = layer(x, pos_offset, layer_kv)
            new_kv_cache.append(new_kv)

        x = self.norm(x)
        # Tied embedding weights as LM head.
        logits = x @ self.embed_tokens.embedding[...].T
        return logits, new_kv_cache
