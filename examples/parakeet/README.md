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
  global `rel_pos` attention. This is what runs on MPS. The relative-position
  attention uses the fused `mps.sdpa` kernel, passing the positional-bias term as
  an additive mask so content+position attention is a single kernel. LayerNorm and
  the biased projection are auto-fused by the plugin (`@mps.layer_norm`, `@mps.addmm`).
- **`decode.py`** — token-and-duration transducer (TDT) greedy decode: prediction
  network (embedding + 2-layer LSTM) + joint network, run host-side in numpy (the
  loop is inherently sequential and the compute is tiny).
- **`main.py`** / **`benchmark.py`** — CLI transcription and an A/B benchmark vs
  `mlx-audio`.

## Run

From this directory (`make` fetches the demo clip on first use):

```bash
make sample       # curl sample.wav (public-domain JFK speech clip, ~11s)
make transcribe   # JAX_PLATFORMS=mps uv run main.py
make benchmark    # parity + RTF vs mlx-audio, swept over float32/bfloat16/float16
make bench-async  # A/B the plugin's async dispatch (JAX_MPS_ASYNC_DISPATCH)
```

Or invoke directly (audio path is a positional argument):

```bash
JAX_PLATFORMS=mps uv run examples/parakeet/main.py examples/parakeet/sample.wav
JAX_PLATFORMS=mps uv run examples/parakeet/benchmark.py examples/parakeet/sample.wav
```

The demo clip is whisper.cpp's public-domain `jfk.wav`; the expected transcript is
`And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.`

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
numbers swing ~2x on Apple Silicon). `mlx-audio` always runs float32 (no dtype knob);
the JAX pipeline is swept across precisions. The numbers below are **indicative** —
a single run on one M-series machine (11 s clip, 20 rounds, min wall); system load and
thermal state move them noticeably, so treat the *ratios* as the signal, not the
absolute ms:

| pipeline | min wall | RTF | vs mlx-audio |
| --- | --- | --- | --- |
| mlx-audio float32 | 169.3 ms | 65.0x | baseline |
| JAX-MPS float32 | 151.1 ms | 72.8x | ~1.2x |
| JAX-MPS bfloat16 | 123.1 ms | 89.4x | **~1.4x** |
| JAX-MPS float16 | 126.1 ms | 87.2x | ~1.3x |

All three precisions produce the identical transcript. The edge comes from (a) fusing
the full relative-position attention into one `mps.sdpa` kernel, (b) running the joint's
encoder projection once outside the sequential decode loop, and (c) dropping to bfloat16
— which mlx-audio can't do out of the box. The interleaved A/B keeps the JAX-vs-mlx
comparison fair under load, but absolute numbers vary; run `make benchmark` yourself.

## Notes / scope

- Greedy TDT decoding, single clip (no beam search, long-audio chunking, or streaming).
- Global `rel_pos` attention only (as used by v2); no custom Metal kernels — every op
  lowers through the plugin's `stablehlo.convolution` / FFT / attention support.
- `float32` is the default and gives exact parity; `--dtype bfloat16` trades a little
  accuracy for speed.
