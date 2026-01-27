# ResNet18 CIFAR-10 Example

Train a ResNet18 model on CIFAR-10 using [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/index.html).

## Usage

```bash
# Train on MPS (GPU)
JAX_PLATFORMS=mps uv run examples/resnet/main.py

# Train on CPU for comparison
JAX_PLATFORMS=cpu uv run examples/resnet/main.py

# Limit training steps
JAX_PLATFORMS=mps uv run examples/resnet/main.py --steps=30
```

## Benchmark

On an M4 MacBook Air, MPS achieves ~3x speedup over CPU:

| Backend | Time per step |
|---------|---------------|
| CPU     | 3.0s          |
| MPS     | 1.0s          |

## Files

- `main.py` - Training loop with Adam optimizer
- `model.py` - ResNet18 architecture adapted for 32x32 images
- `data.py` - CIFAR-10 download and preprocessing
