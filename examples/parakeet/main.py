"""Parakeet-TDT-0.6b speech-to-text using Flax NNX on the JAX MPS backend.

Reimplements NVIDIA's Parakeet-TDT (the model mlx-audio runs) in JAX: log-mel
front-end + Conformer encoder (JIT-compiled onto MPS) + greedy TDT decoder.

    JAX_PLATFORMS=mps uv run examples/parakeet/main.py examples/parakeet/sample.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from audio import PreprocessConfig, build_frontend, load_audio, log_mel
from decode import greedy_decode, load_decoder, load_vocabulary
from flax import nnx
from huggingface_hub import snapshot_download
from model import Conformer, ConformerConfig
from safetensors.numpy import load_file
from weights import load_encoder


class Parakeet:
    def __init__(self, model_id: str, dtype=jnp.float32):
        path = Path(
            snapshot_download(
                model_id,
                allow_patterns=["*.safetensors", "tokenizer.model", "config.json"],
            )
        )
        tensors = {}
        for f in sorted(path.glob("*.safetensors")):
            tensors.update(load_file(str(f)))

        self.preprocess = PreprocessConfig()
        self.dtype = dtype

        # Build the module structure without running (device) initializers, then
        # fill in the real weights -- avoids initializing every param on device.
        encoder = jax.eval_shape(lambda: Conformer(ConformerConfig(dtype=dtype)))
        load_encoder(encoder, tensors, dtype=dtype)
        self._graphdef, self._state = nnx.split(encoder)

        window, filterbank = build_frontend(self.preprocess)
        self.decoder = load_decoder(tensors)
        self.vocab, _ = load_vocabulary(str(path / "tokenizer.model"))

        # One compiled graph from waveform to encoder features (log-mel + encoder).
        # Cast to float32 in-graph so the host copy is a single Execute and the
        # decoder gets float32.
        @jax.jit
        def encode(state, wav):
            mel = log_mel(wav, self.preprocess, window, filterbank)
            return nnx.merge(self._graphdef, state)(mel).astype(jnp.float32)

        self._encode = encode

    def transcribe(self, wav: np.ndarray) -> str:
        feats = np.asarray(self._encode(self._state, jnp.asarray(wav, self.dtype)))
        text, _ = greedy_decode(
            feats, self.decoder, self.vocab, durations=(0, 1, 2, 3, 4)
        )
        return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to an audio file to transcribe")
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    args = parser.parse_args()

    dtype = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}[
        args.dtype
    ]
    print(f"JAX devices: {jax.devices()}")

    model = Parakeet(args.model, dtype=dtype)
    wav = load_audio(args.audio, model.preprocess.sample_rate)
    audio_s = len(wav) / model.preprocess.sample_rate

    # Warm up JIT, then time a run.
    _ = model.transcribe(wav)
    t0 = perf_counter()
    text = model.transcribe(wav)
    dt = perf_counter() - t0

    print(f"\nAudio: {args.audio} ({audio_s:.2f}s)")
    print(f"Transcript: {text}")
    print(f"\n{dt:.3f}s wall ({audio_s / dt:.1f}x realtime)")


if __name__ == "__main__":
    main()
