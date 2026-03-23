# Gemma Inference Example

Generate text with [Gemma](https://ai.google.dev/gemma) using [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/index.html). Weights are loaded directly from HuggingFace Hub.

## Prerequisites

Gemma weights require accepting Google's license. Visit the model page on HuggingFace (e.g., [google/gemma-2b](https://huggingface.co/google/gemma-2b)), accept the license, then authenticate:

```bash
uv run hf auth login
```

## Usage

```bash
# Generate on MPS (GPU)
JAX_PLATFORMS=mps uv run examples/gemma/main.py

# Generate on CPU
JAX_PLATFORMS=cpu uv run examples/gemma/main.py

# Custom prompt and model
JAX_PLATFORMS=mps uv run examples/gemma/main.py --model google/gemma-7b --prompt "Once upon a time"

# Greedy decoding
JAX_PLATFORMS=mps uv run examples/gemma/main.py --temperature 0
```

## Options

| Flag            | Default              | Description                         |
|-----------------|----------------------|-------------------------------------|
| `--model`       | `google/gemma-2b`    | HuggingFace model ID                |
| `--prompt`      | `The meaning of...`  | Input prompt                        |
| `--max-tokens`  | `100`                | Maximum new tokens to generate      |
| `--temperature` | `0.8`                | Sampling temperature (0 = greedy)   |

## Files

- `main.py` - Weight loading, tokenization, and text generation
- `model.py` - Gemma transformer architecture
