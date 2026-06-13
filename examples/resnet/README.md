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

On an M4 MacBook Air, MPS achieves ~4.7x speedup over CPU:

| Backend | Time per step |
|---------|---------------|
| CPU     | 3.2s          |
| MPS     | 0.7s          |

## Async dispatch

`JAX_MPS_ASYNC_DISPATCH=1` lets each step return before the GPU finishes, so the
CPU can queue the next step while the current one runs (CPU/GPU pipelining). It
only helps if the loop avoids a per-step host sync: this example keeps each
step's loss on-device and blocks once at the end of the timing window. Calling
`.item()` every step would serialize CPU and GPU and hide the effect entirely —
making the loop async-friendly is the prerequisite, and on its own the bigger
lever here.

Interleaved A/B on this machine (thermal noise controlled by alternating order;
see `scripts/bench_resnet_async.py`) shows **no reliable steady-state speedup on
ResNet** — two passes bracketed 1.0x (0.98–1.09x). ResNet training is
compute-bound per step (the convolutions dominate GPU time), so overlapping the
next dispatch saves little. The flag pays off for *dispatch-bound* workloads —
many small kernels where per-dispatch overhead rivals the compute — where
`scripts/bench_async_dispatch.py` measures up to ~10x. It is off by default;
enable it per workload and measure.

## Files

- `main.py` - Training loop with Adam optimizer
- `model.py` - ResNet18 architecture adapted for 32x32 images
- `data.py` - CIFAR-10 download and preprocessing
