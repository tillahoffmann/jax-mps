"""Gemma text generation using Flax NNX."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import sentencepiece as spm
from flax import nnx
from huggingface_hub import snapshot_download
from model import Gemma, GemmaConfig
from safetensors.numpy import load_file


def load_model(
    model_id: str, dtype=jnp.float32
) -> tuple[Gemma, spm.SentencePieceProcessor]:
    """Download and load a Gemma model and tokenizer from HuggingFace."""
    path = Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "tokenizer.model", "config.json"],
        )
    )

    # Parse config.
    with open(path / "config.json") as f:
        cfg = json.load(f)
    config = GemmaConfig(
        vocab_size=cfg["vocab_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        dtype=dtype,
    )

    # Create model without allocating weights, then load from safetensors.
    model = jax.eval_shape(lambda: Gemma(config, rngs=nnx.Rngs(0)))
    tensors = {}
    for f in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(str(f)))

    def w(name):
        return jnp.array(tensors[name]).astype(dtype)

    model.embed_tokens.embedding.set_value(w("model.embed_tokens.weight"))
    model.norm.weight.set_value(w("model.norm.weight"))
    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}"
        layer.self_attn.q_proj.kernel.set_value(w(f"{p}.self_attn.q_proj.weight").T)
        layer.self_attn.k_proj.kernel.set_value(w(f"{p}.self_attn.k_proj.weight").T)
        layer.self_attn.v_proj.kernel.set_value(w(f"{p}.self_attn.v_proj.weight").T)
        layer.self_attn.o_proj.kernel.set_value(w(f"{p}.self_attn.o_proj.weight").T)
        gate_w = w(f"{p}.mlp.gate_proj.weight").T
        up_w = w(f"{p}.mlp.up_proj.weight").T
        layer.mlp.gate_up_proj.kernel.set_value(
            jnp.concatenate([gate_w, up_w], axis=-1)
        )
        layer.mlp.down_proj.kernel.set_value(w(f"{p}.mlp.down_proj.weight").T)
        layer.input_layernorm.weight.set_value(w(f"{p}.input_layernorm.weight"))
        layer.post_attention_layernorm.weight.set_value(
            w(f"{p}.post_attention_layernorm.weight")
        )

    # Load tokenizer.
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(path / "tokenizer.model"))

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=100):
    """Greedy autoregressive generation with static KV cache (JIT-compiled)."""
    from time import perf_counter

    graphdef, state = nnx.split(model)
    config = model.config

    @jax.jit
    def prefill(state, input_ids):
        m = nnx.merge(graphdef, state)
        logits, kv_pairs = m(input_ids)
        # Copy prefill KV into static cache (matching dtype).
        static_cache = []
        for k, v in kv_pairs:
            k_pad = jnp.zeros(
                (1, config.num_key_value_heads, max_seq_len, config.head_dim),
                dtype=k.dtype,
            )
            v_pad = jnp.zeros_like(k_pad)
            k_pad = jax.lax.dynamic_update_slice(k_pad, k, (0, 0, 0, 0))
            v_pad = jax.lax.dynamic_update_slice(v_pad, v, (0, 0, 0, 0))
            static_cache.append((k_pad, v_pad))
        return logits, static_cache

    @jax.jit
    def decode_step(state, token, kv_cache, cache_index):
        m = nnx.merge(graphdef, state)
        logits, kv_cache = m(
            token, kv_cache=kv_cache, pos_offset=cache_index, cache_index=cache_index
        )
        return logits, kv_cache

    input_ids = jnp.array([[tokenizer.bos_id()] + tokenizer.Encode(prompt)])
    max_seq_len = input_ids.shape[1] + max_new_tokens
    logits, kv_cache = prefill(state, input_ids)
    pos = input_ids.shape[1]

    t0 = perf_counter()
    tokens_generated = 0
    for _ in range(max_new_tokens):
        next_token = jnp.argmax(logits[0, -1, :])
        if int(next_token) == tokenizer.eos_id():
            break
        print(tokenizer.Decode([int(next_token)]), end="", flush=True)
        logits, kv_cache = decode_step(state, next_token[None, None], kv_cache, pos)
        pos += 1
        tokens_generated += 1
    elapsed = perf_counter() - t0
    print()
    if tokens_generated > 0:
        print(
            f"\n{tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated / elapsed:.1f} tok/s)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2b")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    args = parser.parse_args()

    dtype = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}[
        args.dtype
    ]

    print(f"JAX devices: {jax.devices()}")
    print(f"Dtype: {args.dtype}")
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"Prompt: {args.prompt}")
    generate(model, tokenizer, args.prompt, args.max_tokens)
