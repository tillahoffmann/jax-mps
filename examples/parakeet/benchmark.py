"""Benchmark the JAX Parakeet pipeline against mlx-audio on the same clip.

Verifies token-for-token transcript parity, then reports warm real-time factor
(RTF = audio_seconds / wall_seconds). Runs are interleaved A/B to average out
thermal throttling, which can swing single-shot numbers ~2x on Apple Silicon.

mlx-audio always runs float32 (it has no dtype knob); the JAX pipeline is swept
across float32/bfloat16/float16 so you can see the low-precision speedup and
confirm the transcript still matches.

    JAX_PLATFORMS=mps uv run examples/parakeet/benchmark.py sample.wav
    JAX_PLATFORMS=mps uv run examples/parakeet/benchmark.py sample.wav --dtype bfloat16
"""

from __future__ import annotations

import argparse
import statistics
from time import perf_counter

import jax.numpy as jnp
import soundfile as sf
from audio import load_audio
from main import Parakeet
from mlx_audio.stt.utils import load_model

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


def _rtf(times, audio_s):
    lo = min(times)
    return lo, statistics.median(times), audio_s / lo


def make_jax(audio_path, model_id, dtype_str):
    model = Parakeet(model_id, dtype=DTYPES[dtype_str])
    wav = load_audio(audio_path, model.preprocess.sample_rate)
    model.transcribe(wav)  # warm up JIT
    return lambda: model.transcribe(wav)


def make_mlx(audio_path, model_id):
    model = load_model(model_id)
    model.generate(audio_path)  # warm up  # pyright: ignore[reportOptionalCall]
    return lambda: model.generate(audio_path).text.strip()  # pyright: ignore[reportOptionalCall]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to an audio file to benchmark")
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    parser.add_argument("--dtype", default="all", choices=["all", *DTYPES])
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    info = sf.info(args.audio)
    audio_s = info.frames / info.samplerate
    dtypes = list(DTYPES) if args.dtype == "all" else [args.dtype]

    print(f"Audio: {args.audio} ({audio_s:.2f}s)   rounds={args.rounds}\n")

    mlx_run = make_mlx(args.audio, args.model)
    mlx_text = mlx_run()
    print(f"mlx-audio (float32): {mlx_text}\n")

    print(f"{'pipeline':26s}{'min':>9s}{'med':>9s}{'RTF':>9s}   parity")
    # Time mlx once up front as the shared baseline row.
    mlx_t = []
    for _ in range(args.rounds):
        t0 = perf_counter()
        mlx_run()
        mlx_t.append(perf_counter() - t0)
    lo, md, rtf = _rtf(mlx_t, audio_s)
    print(
        f"{'mlx-audio float32':26s}{lo * 1000:7.1f}ms{md * 1000:7.1f}ms{rtf:8.1f}x   (baseline)"
    )

    for dt in dtypes:
        jax_run = make_jax(args.audio, args.model, dt)
        jax_text = jax_run()
        parity = "MATCH" if jax_text == mlx_text else "MISMATCH"
        jax_t, mx_t = [], []
        for _ in range(args.rounds):  # interleaved A/B vs mlx to share thermal state
            t0 = perf_counter()
            jax_run()
            jax_t.append(perf_counter() - t0)
            t0 = perf_counter()
            mlx_run()
            mx_t.append(perf_counter() - t0)
        lo, md, rtf = _rtf(jax_t, audio_s)
        speed = min(mx_t) / lo
        print(
            f"{'JAX-MPS ' + dt:26s}{lo * 1000:7.1f}ms{md * 1000:7.1f}ms{rtf:8.1f}x   {parity}  ({speed:.2f}x mlx)"
        )


if __name__ == "__main__":
    main()
