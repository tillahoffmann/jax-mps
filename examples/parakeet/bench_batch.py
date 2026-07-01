"""Measure how encoder throughput scales with batch size on MPS.

The Conformer encoder is batch-capable; only variable-length batching needs
length masking (mel normalization + attention/conv), which this script sidesteps
by tiling one clip to batch size B (identical rows -> trivially correct). It
reports amortized per-clip encoder wall time, i.e. the throughput win from
batching many same-length clips.

    JAX_PLATFORMS=mps uv run examples/parakeet/bench_batch.py
    JAX_MPS_ASYNC_DISPATCH=1 JAX_PLATFORMS=mps uv run examples/parakeet/bench_batch.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import jax
    import jax.numpy as jnp
    from audio import load_audio
    from flax import nnx
    from main import Parakeet

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    parser.add_argument("--audio", default=str(Path(__file__).parent / "sample.wav"))
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"]
    )
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[
        args.dtype
    ]
    batches = [int(b) for b in args.batches.split(",")]

    model = Parakeet(args.model, dtype=dtype)
    wav = load_audio(args.audio, model.pre.sample_rate)
    audio_s = len(wav) / model.pre.sample_rate

    mel = model.mel(wav)  # (1, T, 128)
    graphdef, state = model._graphdef, model._state

    @jax.jit
    def encode(state, mel_b):
        return nnx.merge(graphdef, state)(mel_b)

    print(f"dtype={args.dtype}  async={os.environ.get('JAX_MPS_ASYNC_DISPATCH', '0')}")
    print(
        f"{'batch':>5s}{'total ms':>11s}{'per-clip ms':>13s}{'per-clip RTF':>14s}{'speedup':>9s}"
    )

    base = None
    for b in batches:
        mel_b = jnp.repeat(mel, b, axis=0)
        encode(state, mel_b).block_until_ready()  # warm
        ts = []
        for _ in range(args.rounds):
            t0 = perf_counter()
            encode(state, mel_b).block_until_ready()
            ts.append(perf_counter() - t0)
        total = min(ts)
        per = total / b
        if base is None:
            base = per
        print(
            f"{b:5d}{total * 1000:10.1f}m{per * 1000:12.1f}m{audio_s / per:13.1f}x{base / per:8.2f}x"
        )


if __name__ == "__main__":
    main()
