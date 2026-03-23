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
