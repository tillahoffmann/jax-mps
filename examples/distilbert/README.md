# DistilBERT Sentence Encoding Example

Encode sentences into embeddings with [DistilBERT](https://huggingface.co/distilbert-base-uncased) using Flax NNX.

## Usage

```bash
JAX_PLATFORMS=mps uv run examples/distilbert/main.py
JAX_PLATFORMS=mps uv run examples/distilbert/main.py "Hello world" "Hi there" "The weather is nice"
```
