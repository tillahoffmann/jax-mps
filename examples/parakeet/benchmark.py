"""Benchmark the JAX Parakeet pipeline against mlx-audio on the same clip.

Verifies token-for-token transcript parity, then reports warm real-time factor
(RTF = audio_seconds / wall_seconds). Runs are interleaved A/B to average out
thermal throttling, which can swing single-shot numbers ~2x on Apple Silicon.

    JAX_PLATFORMS=mps uv run examples/parakeet/benchmark.py --audio examples/parakeet/sample.wav
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from time import perf_counter


def _fmt(label, ours, audio_s):
    lo = min(ours)
    return f"{label:22s} min {lo * 1000:7.1f}ms  med {statistics.median(ours) * 1000:7.1f}ms  ({audio_s / lo:5.1f}x realtime)"


def bench_jax(audio_path, model_id, dtype_str, rounds):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import jax.numpy as jnp
    from audio import load_audio
    from main import Parakeet

    dtype = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}[
        dtype_str
    ]
    model = Parakeet(model_id, dtype=dtype)
    wav = load_audio(audio_path, model.pre.sample_rate)
    text = model.transcribe(wav)  # warm up + get transcript

    def run():
        return model.transcribe(wav)

    return model, wav, text, run


def bench_mlx(audio_path, model_id):
    from mlx_audio.stt.utils import load_model

    model = load_model(model_id)
    text = model.generate(audio_path).text.strip()  # pyright: ignore[reportOptionalCall]

    def run():
        return model.generate(audio_path).text  # pyright: ignore[reportOptionalCall]

    return text, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    parser.add_argument("--audio", default=str(Path(__file__).parent / "sample.wav"))
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()

    import soundfile as sf

    info = sf.info(args.audio)
    audio_s = info.frames / info.samplerate

    print(
        f"Audio: {args.audio} ({audio_s:.2f}s)   dtype={args.dtype}   rounds={args.rounds}\n"
    )

    _, _, jax_text, jax_run = bench_jax(args.audio, args.model, args.dtype, args.rounds)
    mlx_text, mlx_run = bench_mlx(args.audio, args.model)

    print(f"JAX   transcript: {jax_text}")
    print(f"mlx   transcript: {mlx_text}")
    match = jax_text == mlx_text
    print(f"Transcript parity: {'MATCH' if match else 'MISMATCH'}\n")

    jax_t, mlx_t = [], []
    for _ in range(args.rounds):  # interleaved A/B
        t0 = perf_counter()
        jax_run()
        jax_t.append(perf_counter() - t0)
        t0 = perf_counter()
        mlx_run()
        mlx_t.append(perf_counter() - t0)

    print(_fmt("JAX-MPS (this repo)", jax_t, audio_s))
    print(_fmt("mlx-audio", mlx_t, audio_s))
    speedup = min(mlx_t) / min(jax_t)
    print(f"\nJAX-MPS is {speedup:.2f}x mlx-audio (min wall, warm).")


if __name__ == "__main__":
    main()
