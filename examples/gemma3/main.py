"""Gemma 3 text generation using Flax NNX."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import sentencepiece as spm
from flax import nnx
from huggingface_hub import snapshot_download
from model import Gemma3, Gemma3Config
from safetensors.numpy import load_file


def load_model(
    model_id: str, dtype=jnp.float32
) -> tuple[Gemma3, spm.SentencePieceProcessor]:
    """Download and load a Gemma 3 model and tokenizer from HuggingFace."""
    path = Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "tokenizer.model", "config.json"],
        )
    )

    # Parse config — Gemma 3 4B is multimodal, text config is nested.
    with open(path / "config.json") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)

    # Load weights (need shapes to infer head counts for sparse configs).
    tensors = {}
    for f in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(str(f)))

    # Infer head counts from weight shapes if not in config.
    head_dim = text_cfg.get("head_dim", 256)
    for prefix in ["language_model.model.", ""]:
        q_key = prefix + "layers.0.self_attn.q_proj.weight"
        k_key = prefix + "layers.0.self_attn.k_proj.weight"
        if q_key in tensors and k_key in tensors:
            inferred_n_heads = tensors[q_key].shape[0] // head_dim
            inferred_n_kv_heads = tensors[k_key].shape[0] // head_dim
            break
    else:
        inferred_n_heads = 8
        inferred_n_kv_heads = 4

    # Fill defaults for sparse configs.
    defaults = {
        "num_attention_heads": inferred_n_heads,
        "num_key_value_heads": inferred_n_kv_heads,
        "head_dim": head_dim,
        "query_pre_attn_scalar": head_dim,
        "rope_theta": 1_000_000.0,
        "rope_local_base_freq": 10_000.0,
        "rms_norm_eps": 1e-6,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
        "vocab_size": 262144,
    }
    for k, v in defaults.items():
        if k not in text_cfg:
            text_cfg[k] = v

    config = Gemma3Config(
        vocab_size=text_cfg["vocab_size"],
        num_hidden_layers=text_cfg["num_hidden_layers"],
        hidden_size=text_cfg["hidden_size"],
        intermediate_size=text_cfg["intermediate_size"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        head_dim=text_cfg["head_dim"],
        query_pre_attn_scalar=text_cfg["query_pre_attn_scalar"],
        rope_theta=text_cfg["rope_theta"],
        rope_local_base_freq=text_cfg["rope_local_base_freq"],
        rms_norm_eps=text_cfg["rms_norm_eps"],
        sliding_window=text_cfg["sliding_window"],
        sliding_window_pattern=text_cfg["sliding_window_pattern"],
        dtype=dtype,
    )

    # Create model without allocating weights.
    model = jax.eval_shape(lambda: Gemma3(config, rngs=nnx.Rngs(0)))

    def w(name):
        # Text weights are prefixed with 'language_model.model.' in multimodal checkpoint.
        for prefix in ["language_model.model.", ""]:
            full = prefix + name
            if full in tensors:
                return jnp.array(tensors[full]).astype(dtype)
        raise KeyError(f"Weight not found: {name}")

    # Actual vocab size from weights may differ from config (padding).
    embed_weight = w("embed_tokens.weight")
    actual_vocab = embed_weight.shape[0]
    if actual_vocab != config.vocab_size:
        config.vocab_size = actual_vocab
        model = jax.eval_shape(lambda: Gemma3(config, rngs=nnx.Rngs(0)))

    model.embed_tokens.embedding.set_value(embed_weight)
    model.norm.weight.set_value(w("norm.weight"))

    for i, layer in enumerate(model.layers):
        p = f"layers.{i}"
        # Attention projections.
        layer.self_attn.q_proj.kernel.set_value(w(f"{p}.self_attn.q_proj.weight").T)
        layer.self_attn.k_proj.kernel.set_value(w(f"{p}.self_attn.k_proj.weight").T)
        layer.self_attn.v_proj.kernel.set_value(w(f"{p}.self_attn.v_proj.weight").T)
        layer.self_attn.o_proj.kernel.set_value(w(f"{p}.self_attn.o_proj.weight").T)
        # QK norms.
        layer.self_attn.q_norm.weight.set_value(w(f"{p}.self_attn.q_norm.weight"))
        layer.self_attn.k_norm.weight.set_value(w(f"{p}.self_attn.k_norm.weight"))
        # MLP — fused gate+up.
        gate_w = w(f"{p}.mlp.gate_proj.weight").T
        up_w = w(f"{p}.mlp.up_proj.weight").T
        layer.mlp.gate_up_proj.kernel.set_value(
            jnp.concatenate([gate_w, up_w], axis=-1)
        )
        layer.mlp.down_proj.kernel.set_value(w(f"{p}.mlp.down_proj.weight").T)
        # Layer norms (4 per layer in Gemma 3).
        layer.input_layernorm.weight.set_value(w(f"{p}.input_layernorm.weight"))
        layer.post_attention_layernorm.weight.set_value(
            w(f"{p}.post_attention_layernorm.weight")
        )
        layer.pre_feedforward_layernorm.weight.set_value(
            w(f"{p}.pre_feedforward_layernorm.weight")
        )
        layer.post_feedforward_layernorm.weight.set_value(
            w(f"{p}.post_feedforward_layernorm.weight")
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

    input_ids = jnp.array([[tokenizer.bos_id()] + tokenizer.Encode(prompt)])
    max_seq_len = input_ids.shape[1] + max_new_tokens

    # Determine cache size per layer: sliding layers use a rotating cache
    # of sliding_window size, global layers use the full max_seq_len.
    cache_sizes = []
    for i in range(config.num_hidden_layers):
        is_sliding = (i + 1) % config.sliding_window_pattern != 0
        if is_sliding and config.sliding_window < max_seq_len:
            cache_sizes.append(config.sliding_window)
        else:
            cache_sizes.append(max_seq_len)

    @jax.jit
    def prefill(state, input_ids):
        m = nnx.merge(graphdef, state)
        logits, kv_pairs = m(input_ids)
        # Copy prefill KV into static cache (matching dtype).
        static_cache = []
        for i, (k, v) in enumerate(kv_pairs):
            cs = cache_sizes[i]
            k_pad = jnp.zeros(
                (1, config.num_key_value_heads, cs, config.head_dim),
                dtype=k.dtype,
            )
            v_pad = jnp.zeros_like(k_pad)
            # For rotating caches, only copy the last sliding_window positions.
            if cs < max_seq_len and k.shape[2] > cs:
                k = k[:, :, -cs:, :]
                v = v[:, :, -cs:, :]
            k_pad = jax.lax.dynamic_update_slice(k_pad, k, (0, 0, 0, 0))
            v_pad = jax.lax.dynamic_update_slice(v_pad, v, (0, 0, 0, 0))
            static_cache.append((k_pad, v_pad))
        next_token = jnp.argmax(logits[0, -1, :], keepdims=True)
        return next_token, static_cache

    @jax.jit
    def decode_step(state, token, kv_cache, cache_index):
        m = nnx.merge(graphdef, state)
        logits, kv_cache = m(
            token, kv_cache=kv_cache, pos_offset=cache_index, cache_index=cache_index
        )
        next_token = jnp.argmax(logits[0, -1, :], keepdims=True)
        return next_token, kv_cache

    token, kv_cache = prefill(state, input_ids)
    pos = input_ids.shape[1]

    eos_id = tokenizer.eos_id()
    generated_ids = []

    t0 = perf_counter()
    for _ in range(max_new_tokens):
        # Block on current token to check for EOS.
        tok_id = int(token[0])
        if tok_id == eos_id:
            break
        generated_ids.append(tok_id)
        # Dispatch next decode step.
        token, kv_cache = decode_step(state, token[None], kv_cache, pos)
        pos += 1
    # Sync to include any in-flight computation in the timing.
    _ = token.block_until_ready()
    elapsed = perf_counter() - t0

    tokens_generated = len(generated_ids)
    print(tokenizer.Decode(generated_ids))
    if tokens_generated > 0:
        print(
            f"\n{tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated / elapsed:.1f} tok/s)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
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

    # Gemma 3's RMSNorm overflows in float16 on CPU (pure-JAX fallback
    # computes x^2 in the input dtype). Promote to float32 automatically.
    if args.dtype == "float16" and all(d.platform == "cpu" for d in jax.devices()):
        print("Warning: float16 on CPU overflows for Gemma 3; promoting to float32.")
        dtype = jnp.float32

    print(f"JAX devices: {jax.devices()}")
    print(f"Dtype: {dtype}")
    model, tokenizer = load_model(args.model, dtype=dtype)
    print(f"Prompt: {args.prompt}")
    generate(model, tokenizer, args.prompt, args.max_tokens)
