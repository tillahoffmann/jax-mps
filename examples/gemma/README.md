# Gemma Inference Example

Generate text with [Gemma](https://ai.google.dev/gemma) using Flax NNX.

## Setup

Accept the license at [google/gemma-2b](https://huggingface.co/google/gemma-2b), then:

```bash
uv run hf auth login
```

## Usage

```bash
JAX_PLATFORMS=mps uv run examples/gemma/main.py
JAX_PLATFORMS=mps uv run examples/gemma/main.py --model google/gemma-7b --prompt "Once upon a time"
```

Once the weights are cached, add `HF_HUB_OFFLINE=1` to skip the Hugging Face Hub
revision check (which otherwise blocks on the network before every run):

```bash
HF_HUB_OFFLINE=1 JAX_PLATFORMS=mps uv run examples/gemma/main.py
```

## Weight-only quantization

Autoregressive decode is memory-bandwidth-bound: each token reads ~all the
weights once. Quantizing those weights to int4/int8 (via the fused
`mps.quantized_matmul` kernel) cuts that traffic and speeds up decode. Two opt-in
flags:

```bash
# int8 LM head (~21% of the per-token weight read)
HF_HUB_OFFLINE=1 JAX_PLATFORMS=mps uv run examples/gemma/main.py --quantize-lm-head

# MLP dense layers (~72%): int8, or int4 (which uses a mixed recipe --
# gate_up int4, the more-sensitive down_proj int8)
... --quantize-mlp-bits 8
... --quantize-mlp-bits 4

# both together
... --quantize-lm-head --quantize-mlp-bits 4
```

### Measured (gemma-2b, fp16 decode, M-series Apple Silicon, cool machine)

| Config | tok/s | vs fp16 |
|---|---|---|
| fp16 baseline | 14.2 | — |
| `--quantize-lm-head` (int8) | 16.1 | +13% |
| `--quantize-mlp-bits 8` | 21.6 | +52% |
| `--quantize-lm-head --quantize-mlp-bits 4` | 28.4 | ~2.0x |

Absolute throughput is hardware- and thermal-dependent (sustained runs throttle
and drop the numbers), but the ordering is stable: more quantization -> faster
decode. Quality holds up -- MLP int8 is token-identical to fp16; uniform MLP int4
degrades (repetition), which is why int4 keeps `down_proj` at int8; the full
stack stays coherent and tracks the fp16 base model.
