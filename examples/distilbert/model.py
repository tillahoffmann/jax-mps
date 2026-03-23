"""DistilBERT encoder architecture in Flax NNX."""

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass
class DistilBertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 6
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12


class DistilBertAttention(nnx.Module):
    def __init__(self, config, *, rngs):
        h = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = h // config.num_heads
        # Fused QKV projection: one matmul instead of three.
        self.qkv_lin = nnx.Linear(h, 3 * h, rngs=rngs)
        self.out_lin = nnx.Linear(h, h, rngs=rngs)

    def __call__(self, x, mask):
        B, T, _ = x.shape
        qkv = self.qkv_lin(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, T, N, H) layout for jax.nn.dot_product_attention.
        out = jax.nn.dot_product_attention(q, k, v, mask=mask)
        return self.out_lin(out.reshape(B, T, -1))


class DistilBertFFN(nnx.Module):
    def __init__(self, config, *, rngs):
        self.lin1 = nnx.Linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.lin2 = nnx.Linear(config.intermediate_size, config.hidden_size, rngs=rngs)

    def __call__(self, x):
        return self.lin2(jax.nn.gelu(self.lin1(x), approximate=True))


class DistilBertLayer(nnx.Module):
    def __init__(self, config, *, rngs):
        self.attention = DistilBertAttention(config, rngs=rngs)
        self.sa_layer_norm = nnx.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs
        )
        self.ffn = DistilBertFFN(config, rngs=rngs)
        self.output_layer_norm = nnx.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs
        )

    def __call__(self, x, mask):
        x = self.sa_layer_norm(x + self.attention(x, mask))
        x = self.output_layer_norm(x + self.ffn(x))
        return x


class DistilBert(nnx.Module):
    def __init__(self, config, *, rngs):
        self.word_embeddings = nnx.Embed(
            config.vocab_size, config.hidden_size, rngs=rngs
        )
        self.position_embeddings = nnx.Embed(
            config.max_position_embeddings, config.hidden_size, rngs=rngs
        )
        self.embeddings_layer_norm = nnx.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs
        )
        self.layers = nnx.List(
            DistilBertLayer(config, rngs=rngs) for _ in range(config.num_layers)
        )

    def __call__(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        positions = jnp.arange(T)[None, :]
        x = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embeddings_layer_norm(x)

        # Attention mask: (B, 1, 1, T) boolean for dot_product_attention.
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].astype(bool)
        else:
            mask = None

        for layer in self.layers:
            x = layer(x, mask)
        return x
