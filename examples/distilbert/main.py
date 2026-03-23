"""DistilBERT sentence encoding example."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
from model import DistilBert, DistilBertConfig
from safetensors.numpy import load_file
from tokenizers import Tokenizer


def load_model(model_id: str) -> tuple[DistilBert, Tokenizer]:
    """Download and load a DistilBERT model and tokenizer."""
    path = Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "config.json", "tokenizer.json"],
        )
    )

    with open(path / "config.json") as f:
        cfg = json.load(f)
    config = DistilBertConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg.get("dim", cfg.get("hidden_size", 768)),
        num_heads=cfg.get("n_heads", cfg.get("num_attention_heads", 12)),
        num_layers=cfg.get("n_layers", cfg.get("num_hidden_layers", 6)),
        intermediate_size=cfg.get("hidden_dim", cfg.get("intermediate_size", 3072)),
        max_position_embeddings=cfg.get("max_position_embeddings", 512),
    )

    model = jax.eval_shape(lambda: DistilBert(config, rngs=nnx.Rngs(0)))
    tensors = {}
    for f in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(str(f)))

    p = "distilbert"
    model.word_embeddings.embedding.set_value(
        jnp.array(tensors[f"{p}.embeddings.word_embeddings.weight"])
    )
    model.position_embeddings.embedding.set_value(
        jnp.array(tensors[f"{p}.embeddings.position_embeddings.weight"])
    )
    model.embeddings_layer_norm.scale.set_value(
        jnp.array(tensors[f"{p}.embeddings.LayerNorm.weight"])
    )
    model.embeddings_layer_norm.bias.set_value(
        jnp.array(tensors[f"{p}.embeddings.LayerNorm.bias"])
    )

    for i, layer in enumerate(model.layers):
        lp = f"{p}.transformer.layer.{i}"
        layer.attention.q_lin.kernel.set_value(
            jnp.array(tensors[f"{lp}.attention.q_lin.weight"]).T
        )
        layer.attention.q_lin.bias.set_value(
            jnp.array(tensors[f"{lp}.attention.q_lin.bias"])
        )
        layer.attention.k_lin.kernel.set_value(
            jnp.array(tensors[f"{lp}.attention.k_lin.weight"]).T
        )
        layer.attention.k_lin.bias.set_value(
            jnp.array(tensors[f"{lp}.attention.k_lin.bias"])
        )
        layer.attention.v_lin.kernel.set_value(
            jnp.array(tensors[f"{lp}.attention.v_lin.weight"]).T
        )
        layer.attention.v_lin.bias.set_value(
            jnp.array(tensors[f"{lp}.attention.v_lin.bias"])
        )
        layer.attention.out_lin.kernel.set_value(
            jnp.array(tensors[f"{lp}.attention.out_lin.weight"]).T
        )
        layer.attention.out_lin.bias.set_value(
            jnp.array(tensors[f"{lp}.attention.out_lin.bias"])
        )
        layer.sa_layer_norm.scale.set_value(
            jnp.array(tensors[f"{lp}.sa_layer_norm.weight"])
        )
        layer.sa_layer_norm.bias.set_value(
            jnp.array(tensors[f"{lp}.sa_layer_norm.bias"])
        )
        layer.ffn.lin1.kernel.set_value(jnp.array(tensors[f"{lp}.ffn.lin1.weight"]).T)
        layer.ffn.lin1.bias.set_value(jnp.array(tensors[f"{lp}.ffn.lin1.bias"]))
        layer.ffn.lin2.kernel.set_value(jnp.array(tensors[f"{lp}.ffn.lin2.weight"]).T)
        layer.ffn.lin2.bias.set_value(jnp.array(tensors[f"{lp}.ffn.lin2.bias"]))
        layer.output_layer_norm.scale.set_value(
            jnp.array(tensors[f"{lp}.output_layer_norm.weight"])
        )
        layer.output_layer_norm.bias.set_value(
            jnp.array(tensors[f"{lp}.output_layer_norm.bias"])
        )

    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    return model, tokenizer


def encode(model, tokenizer, sentences, batch_size=32):
    """Encode sentences into [CLS] embeddings."""
    encoded = tokenizer.encode_batch(sentences)
    all_ids = [e.ids for e in encoded]
    all_masks = [e.attention_mask for e in encoded]

    # Pad to max length in batch.
    max_len = max(len(ids) for ids in all_ids)
    input_ids = jnp.array([ids + [0] * (max_len - len(ids)) for ids in all_ids])
    attention_mask = jnp.array([m + [0] * (max_len - len(m)) for m in all_masks])

    # Run model and extract [CLS] token embeddings.
    hidden = model(input_ids, attention_mask)
    return hidden[:, 0, :]  # [CLS] is always position 0


def cosine_similarity(a, b):
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument(
        "sentences",
        nargs="*",
        default=[
            "The cat sat on the mat.",
            "A kitten was sitting on a rug.",
            "The stock market crashed today.",
        ],
    )
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    model, tokenizer = load_model(args.model)

    # Warmup.
    encode(model, tokenizer, ["warmup"])

    t0 = perf_counter()
    embeddings = encode(model, tokenizer, args.sentences)
    elapsed = perf_counter() - t0

    print(f"\nEncoded {len(args.sentences)} sentences in {elapsed:.3f}s")
    print(f"Embedding shape: {embeddings.shape}")

    # Show pairwise cosine similarities.
    print("\nCosine similarities:")
    for i in range(len(args.sentences)):
        for j in range(i + 1, len(args.sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  [{i}] vs [{j}]: {float(sim):.4f}")
            print(f'    "{args.sentences[i]}"')
            print(f'    "{args.sentences[j]}"')
