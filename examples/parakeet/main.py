"""Parakeet-TDT-0.6b speech-to-text using Flax NNX on the JAX MPS backend.

Reimplements NVIDIA's Parakeet-TDT (the model mlx-audio runs) in JAX: log-mel
front-end + Conformer encoder (JIT-compiled onto MPS) + greedy TDT decoder.

    JAX_PLATFORMS=mps uv run examples/parakeet/main.py --audio examples/parakeet/sample.wav
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

        self.cfg = ConformerConfig(dtype=dtype)
        self.pre = PreprocessConfig()
        self.dtype = dtype

        encoder = Conformer(self.cfg)
        load_encoder(encoder, tensors, dtype=dtype)
        self._graphdef, self._state = nnx.split(encoder)

        self.window, self.filterbank = build_frontend(self.pre)
        self.decoder = load_decoder(tensors)
        self.vocab, _ = load_vocabulary(str(path / "tokenizer.model"))

        @jax.jit
        def _encode(state, mel):
            return nnx.merge(self._graphdef, state)(mel)

        # Fused front-end + encoder: one compiled graph from waveform to features.
        @jax.jit
        def _encode_wav(state, wav):
            mel = log_mel(wav, self.pre, self.window, self.filterbank)
            return nnx.merge(self._graphdef, state)(mel)

        self._encode = _encode
        self._encode_wav = _encode_wav

    def mel(self, wav: np.ndarray) -> jnp.ndarray:
        return log_mel(
            jnp.asarray(wav, self.dtype), self.pre, self.window, self.filterbank
        )

    def encode(self, mel: jnp.ndarray) -> np.ndarray:
        feats = self._encode(self._state, mel)
        return np.asarray(feats.astype(jnp.float32))

    def encode_wav(self, wav: np.ndarray) -> np.ndarray:
        feats = self._encode_wav(self._state, jnp.asarray(wav, self.dtype))
        return np.asarray(feats.astype(jnp.float32))

    def transcribe(self, wav: np.ndarray) -> str:
        feats = self.encode_wav(wav)
        text, _ = greedy_decode(
            feats, self.decoder, self.vocab, durations=(0, 1, 2, 3, 4)
        )
        return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    parser.add_argument("--audio", default=str(Path(__file__).parent / "sample.wav"))
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    args = parser.parse_args()

    dtype = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}[
        args.dtype
    ]
    print(f"JAX devices: {jax.devices()}")

    model = Parakeet(args.model, dtype=dtype)
    wav = load_audio(args.audio, model.pre.sample_rate)
    audio_s = len(wav) / model.pre.sample_rate

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
