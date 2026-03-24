"""Benchmark: jax-mps Gemma vs mlx-lm Gemma on the same prompt."""

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "google/gemma-2b"
PROMPT = "The meaning of life is"
MAX_TOKENS = 100


def get_model_path():
    """Get local path for gemma-2b (uses HF cache)."""
    return Path(
        snapshot_download(
            MODEL_ID,
            allow_patterns=["*.safetensors", "tokenizer.model", "config.json"],
        )
    )


def benchmark_mlx_lm(model_path: Path):
    """Run mlx-lm generation and return timing info."""
    try:
        import mlx.core as mx
        import sentencepiece as spm
        from mlx_lm.models.gemma import Model, ModelArgs
    except ImportError as e:
        raise SystemExit(
            "mlx-lm benchmark requires 'mlx', 'mlx_lm', and 'sentencepiece' "
            f"packages on Apple Silicon/macOS. Unable to import: {e}"
        )

    print("=== mlx-lm (native MLX) ===")

    # Load config.
    with open(model_path / "config.json") as f:
        cfg = json.load(f)
    cfg["model_type"] = "gemma"
    args = ModelArgs.from_dict(cfg)

    # Build model.
    model = Model(args)

    # Load weights from safetensors using MLX's loader.
    weights: dict[str, mx.array] = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        loaded = mx.load(str(sf))
        assert isinstance(loaded, dict)
        weights.update(loaded)

    # Map HF weight names to mlx-lm names.
    mapped = {}
    for k, v in weights.items():
        new_k = k
        # model.layers.X.self_attn -> model.layers.X.self_attn
        # model.layers.X.mlp.gate_proj -> model.layers.X.mlp.gate_proj
        # Weights are stored as "weight" in HF but mlx uses them directly.
        new_k = new_k.replace(".weight", "")
        # Linear layers: HF stores (out, in), MLX expects (out, in) too but as .weight
        mapped[new_k + ".weight"] = v

    model.load_weights(list(mapped.items()), strict=False)
    mx.eval(model.parameters())

    # Load tokenizer.
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(model_path / "tokenizer.model"))

    # Simple generate function matching our jax-mps approach.
    def generate_mlx(prompt_text, max_tokens):
        from mlx_lm.models import cache

        input_ids = [tokenizer.bos_id()] + tokenizer.Encode(prompt_text)
        prompt_arr = mx.array(input_ids)

        kv_cache = cache.make_prompt_cache(model)

        # Prefill.
        logits = model(prompt_arr[None], cache=kv_cache)
        mx.eval([c.state for c in kv_cache])

        token = mx.argmax(logits[:, -1, :], axis=-1)
        tokens_out = []

        t0 = time.perf_counter()
        for _ in range(max_tokens):
            mx.eval(token)
            t = token.item()
            if t == tokenizer.eos_id():
                break
            tokens_out.append(t)
            logits = model(token.reshape(1, 1), cache=kv_cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)

        mx.eval(token)
        elapsed = time.perf_counter() - t0
        text = tokenizer.Decode(tokens_out)
        return text, tokens_out, elapsed

    # Warmup.
    print("Warmup...")
    generate_mlx(PROMPT, 5)

    # Timed run.
    print(f"Prompt: {PROMPT}")
    text, tokens, elapsed = generate_mlx(PROMPT, MAX_TOKENS)
    n = len(tokens)
    tps = n / elapsed if elapsed > 0 else 0
    print(f"Generated: {text}")
    print(f"mlx-lm: {n} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)\n")
    return {"name": "mlx-lm", "tokens": n, "elapsed": elapsed, "tps": tps}


def benchmark_jax_mps(model_path: Path, dtype_str="float32"):
    """Run our jax-mps Gemma generation and return timing info."""
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    if not any(d.platform == "mps" for d in devices):
        raise RuntimeError(
            f"benchmark_jax_mps requires an MPS device, but found: {devices}"
        )

    from main import generate as jax_generate
    from main import load_model

    dtype = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}[
        dtype_str
    ]

    print(f"=== jax-mps (dtype={dtype_str}) ===")
    print(f"JAX devices: {jax.devices()}")
    model, tokenizer = load_model(MODEL_ID, dtype=dtype)

    # Warmup (JIT compilation).
    print("Warmup...")
    jax_generate(model, tokenizer, PROMPT, max_new_tokens=5)

    # Timed run.
    print(f"Prompt: {PROMPT}")
    t0 = time.perf_counter()
    jax_generate(model, tokenizer, PROMPT, max_new_tokens=MAX_TOKENS)
    elapsed = time.perf_counter() - t0

    print(f"jax-mps: end-to-end {elapsed:.2f}s for up to {MAX_TOKENS} tokens\n")
    return {"name": "jax-mps", "elapsed": elapsed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--only", choices=["mlx", "jax"], default=None)
    args = parser.parse_args()

    model_path = get_model_path()
    print(f"Model path: {model_path}\n")

    results = []

    if args.only != "jax":
        results.append(benchmark_mlx_lm(model_path))

    if args.only != "mlx":
        results.append(benchmark_jax_mps(model_path, args.dtype))

    print("=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    for r in results:
        if "tps" in r:
            print(f"  {r['name']}: {r['tps']:.1f} tok/s ({r['elapsed']:.2f}s)")
        else:
            print(f"  {r['name']}: {r['elapsed']:.2f}s end-to-end")
