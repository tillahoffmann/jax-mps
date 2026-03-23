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


def load_model(model_id: str) -> tuple[Gemma, spm.SentencePieceProcessor]:
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
    )

    # Create model without allocating weights, then load from safetensors.
    model = jax.eval_shape(lambda: Gemma(config, rngs=nnx.Rngs(0)))
    tensors = {}
    for f in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(str(f)))
    model.embed_tokens.embedding.set_value(
        jnp.array(tensors["model.embed_tokens.weight"])
    )
    model.norm.weight.set_value(jnp.array(tensors["model.norm.weight"]))
    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}"
        layer.self_attn.q_proj.kernel.set_value(
            jnp.array(tensors[f"{p}.self_attn.q_proj.weight"]).T
        )
        layer.self_attn.k_proj.kernel.set_value(
            jnp.array(tensors[f"{p}.self_attn.k_proj.weight"]).T
        )
        layer.self_attn.v_proj.kernel.set_value(
            jnp.array(tensors[f"{p}.self_attn.v_proj.weight"]).T
        )
        layer.self_attn.o_proj.kernel.set_value(
            jnp.array(tensors[f"{p}.self_attn.o_proj.weight"]).T
        )
        layer.mlp.gate_proj.kernel.set_value(
            jnp.array(tensors[f"{p}.mlp.gate_proj.weight"]).T
        )
        layer.mlp.up_proj.kernel.set_value(
            jnp.array(tensors[f"{p}.mlp.up_proj.weight"]).T
        )
        layer.mlp.down_proj.kernel.set_value(
            jnp.array(tensors[f"{p}.mlp.down_proj.weight"]).T
        )
        layer.input_layernorm.weight.set_value(
            jnp.array(tensors[f"{p}.input_layernorm.weight"])
        )
        layer.post_attention_layernorm.weight.set_value(
            jnp.array(tensors[f"{p}.post_attention_layernorm.weight"])
        )

    # Load tokenizer.
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(path / "tokenizer.model"))

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=100):
    """Greedy autoregressive generation with KV cache."""
    input_ids = jnp.array([[tokenizer.bos_id()] + tokenizer.Encode(prompt)])
    logits, kv_cache = model(input_ids)
    pos = input_ids.shape[1]

    for _ in range(max_new_tokens):
        next_token = jnp.argmax(logits[0, -1, :])
        if int(next_token) == tokenizer.eos_id():
            break
        print(tokenizer.Decode([int(next_token)]), end="", flush=True)
        logits, kv_cache = model(
            next_token[None, None], kv_cache=kv_cache, pos_offset=pos
        )
        pos += 1
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2b")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    model, tokenizer = load_model(args.model)
    print(f"Prompt: {args.prompt}")
    generate(model, tokenizer, args.prompt, args.max_tokens)
