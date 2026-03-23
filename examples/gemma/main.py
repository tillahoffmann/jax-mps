"""Gemma text generation using Flax NNX."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import sentencepiece as spm
from dotenv import load_dotenv
from flax import nnx
from huggingface_hub import snapshot_download
from model import Gemma, GemmaConfig
from safetensors.numpy import load_file


def download_model(model_id: str) -> Path:
    """Download model files from HuggingFace Hub."""
    path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "tokenizer.model", "config.json"],
    )
    return Path(path)


def load_config(model_path: Path) -> GemmaConfig:
    """Parse config.json into a GemmaConfig."""
    with open(model_path / "config.json") as f:
        cfg = json.load(f)
    return GemmaConfig(
        vocab_size=cfg["vocab_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        max_position_embeddings=cfg.get("max_position_embeddings", 8192),
        rope_theta=cfg.get("rope_theta", 10000.0),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
    )


def load_weights(model: Gemma, model_path: Path) -> None:
    """Load pretrained weights from safetensors files into the model."""
    tensors = {}
    for f in sorted(model_path.glob("*.safetensors")):
        tensors.update(load_file(str(f)))

    def to_jax(x):
        return jnp.array(x)

    model.embed_tokens.embedding.set_value(to_jax(tensors["model.embed_tokens.weight"]))
    model.norm.weight.set_value(to_jax(tensors["model.norm.weight"]))

    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}"
        layer.self_attn.q_proj.kernel.set_value(
            to_jax(tensors[f"{p}.self_attn.q_proj.weight"]).T
        )
        layer.self_attn.k_proj.kernel.set_value(
            to_jax(tensors[f"{p}.self_attn.k_proj.weight"]).T
        )
        layer.self_attn.v_proj.kernel.set_value(
            to_jax(tensors[f"{p}.self_attn.v_proj.weight"]).T
        )
        layer.self_attn.o_proj.kernel.set_value(
            to_jax(tensors[f"{p}.self_attn.o_proj.weight"]).T
        )
        layer.mlp.gate_proj.kernel.set_value(
            to_jax(tensors[f"{p}.mlp.gate_proj.weight"]).T
        )
        layer.mlp.up_proj.kernel.set_value(to_jax(tensors[f"{p}.mlp.up_proj.weight"]).T)
        layer.mlp.down_proj.kernel.set_value(
            to_jax(tensors[f"{p}.mlp.down_proj.weight"]).T
        )
        layer.input_layernorm.weight.set_value(
            to_jax(tensors[f"{p}.input_layernorm.weight"])
        )
        layer.post_attention_layernorm.weight.set_value(
            to_jax(tensors[f"{p}.post_attention_layernorm.weight"])
        )


def generate(
    model: Gemma,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
) -> str:
    """Generate text autoregressively with KV cache, printing tokens as they appear."""
    input_ids = jnp.array([[tokenizer.bos_id()] + tokenizer.Encode(prompt)])
    eos_id = tokenizer.eos_id()
    key = jax.random.key(0)
    num_generated = 0
    t0 = perf_counter()
    ttft = 0.0
    all_ids = input_ids[0].tolist()

    # Prefill: process entire prompt, get KV cache.
    logits, kv_cache = model(input_ids)
    next_logits = logits[0, -1, :]
    pos = input_ids.shape[1]

    for i in range(max_new_tokens):
        if temperature <= 0:
            next_token = jnp.argmax(next_logits)
        else:
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, next_logits / temperature)

        next_id = int(next_token)

        if i == 0:
            ttft = perf_counter() - t0
            print(f"[time to first token: {ttft:.2f}s]")
            print("Response: ", end="", flush=True)

        if next_id == eos_id:
            break

        num_generated += 1
        all_ids.append(next_id)
        print(tokenizer.Decode([next_id]), end="", flush=True)

        # Decode step: process only the new token with KV cache.
        next_logits, kv_cache = model(
            next_token[None, None], kv_cache=kv_cache, pos_offset=pos
        )
        next_logits = next_logits[0, -1, :]
        pos += 1

    elapsed = perf_counter() - t0
    if num_generated > 0:
        gen_time = elapsed - ttft
        tps = num_generated / gen_time if gen_time > 0 else float("inf")
        print(
            f"\n\n--- {num_generated} tokens in {elapsed:.2f}s "
            f"(TTFT: {ttft:.2f}s, then {tps:.1f} tok/s) ---"
        )
    else:
        print(f"\n\n--- 0 tokens generated in {elapsed:.2f}s ---")
    return tokenizer.Decode(all_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text with Gemma.")
    parser.add_argument(
        "--model",
        default="google/gemma-2b",
        help="HuggingFace model ID (default: google/gemma-2b)",
    )
    parser.add_argument(
        "--prompt",
        default="The meaning of life is",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature; 0 for greedy (default: 0.8)",
    )
    args = parser.parse_args()
    load_dotenv()

    print(f"JAX devices: {jax.devices()}")

    # Download model from HuggingFace Hub.
    print(f"Downloading {args.model}...")
    model_path = download_model(args.model)

    # Build model and load weights.
    config = load_config(model_path)
    print(
        f"Loading model ({config.num_hidden_layers} layers, "
        f"hidden_size={config.hidden_size})..."
    )
    model = jax.eval_shape(lambda: Gemma(config, rngs=nnx.Rngs(0)))
    load_weights(model, model_path)

    # Load tokenizer.
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path / "tokenizer.model"))

    # Generate.
    print(f"\nPrompt: {args.prompt}")
    generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
