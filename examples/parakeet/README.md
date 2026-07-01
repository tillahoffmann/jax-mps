# Parakeet-TDT speech-to-text (JAX / MPS)

A JAX + Flax NNX reimplementation of NVIDIA's **Parakeet-TDT-0.6b-v2** ASR model — the
same model [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) runs — with the
compute-heavy Conformer encoder JIT-compiled onto the **MPS** backend. It loads the
public `mlx-community/parakeet-tdt-0.6b-v2` weights directly and produces transcripts
that match `mlx-audio` token-for-token.

## Pipeline

```
waveform ─► log-mel front-end ─► Conformer encoder ─► TDT greedy decoder ─► text
           (STFT/rfft, mel,      (24 layers, d=1024,   (2-layer LSTM +
            per-feature norm)      rel-pos attention)    joint net, host-side)
```

- **`audio.py`** — log-mel front-end (STFT via `jnp.fft.rfft`, slaney mel filterbank,
  per-feature normalization). Matches NeMo/mlx-audio preprocessing.
- **`model.py`** — FastConformer encoder in NNX: depthwise-striding subsampling
  (×8), 24 Conformer blocks (FFN · relative-position MHSA · conv module · FFN),
  global `rel_pos` attention. This is what runs on MPS.
- **`decode.py`** — token-and-duration transducer (TDT) greedy decode: prediction
  network (embedding + 2-layer LSTM) + joint network, run host-side in numpy (the
  loop is inherently sequential and the compute is tiny).
- **`main.py`** / **`benchmark.py`** — CLI transcription and an A/B benchmark vs
  `mlx-audio`.

## Run

From this directory (`make` fetches the demo clip on first use):

```bash
make sample      # download sample.wav (LibriSpeech "Mister Quilter ...", 5.86s)
make transcribe  # JAX_PLATFORMS=mps uv run main.py
make benchmark   # parity check + RTF vs mlx-audio (ROUNDS=20 by default)
```

Or invoke directly:

```bash
JAX_PLATFORMS=mps uv run examples/parakeet/main.py --audio examples/parakeet/sample.wav
JAX_PLATFORMS=mps uv run examples/parakeet/benchmark.py
```

The demo clip is fetched from `hf-internal-testing/librispeech_asr_dummy`; the expected
transcript is
`mister Quilter is the Apostle of the Middle Classes, and we are glad to welcome his gospel.`

## Correctness

Built and verified bottom-up against `mlx-audio` in float32:

| Stage | Check | Result |
| --- | --- | --- |
| log-mel front-end | max abs diff vs mlx-audio | ~3.6e-4 |
| Conformer encoder | max abs diff vs mlx-audio | ~6.3e-6 |
| full transcript | token-for-token | exact match |

## Performance

`benchmark.py` reports warm real-time factor (RTF = audio seconds / wall seconds),
interleaving JAX and mlx-audio runs A/B to average out thermal throttling (single-shot
numbers swing ~2x on Apple Silicon). On an idle machine (float32, 5.86 s clip, 20 rounds):

| | min wall | RTF |
| --- | --- | --- |
| JAX-MPS (this repo) | 112.0 ms | 52.3x realtime |
| mlx-audio | 112.9 ms | 51.9x realtime |

i.e. the same throughput as `mlx-audio` (1.01x). The encoder dominates; running the
joint's encoder projection once (outside the sequential decode loop) rather than
per-step closes most of the gap. Numbers vary with machine and thermal state — run the
benchmark yourself for a current figure.

## Notes / scope

- Greedy TDT decoding, single clip (no beam search, long-audio chunking, or streaming).
- Global `rel_pos` attention only (as used by v2); no custom Metal kernels — every op
  lowers through the plugin's `stablehlo.convolution` / FFT / attention support.
- `float32` is the default and gives exact parity; `--dtype bfloat16` trades a little
  accuracy for speed.
