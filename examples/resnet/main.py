"""
ResNet18 training on CIFAR-10 using Flax NNX.
"""

import argparse
from time import perf_counter

import jax
import jax.numpy as jnp
import optax
from data import load_cifar10
from flax import nnx
from model import ResNet18
from tqdm import tqdm

# Hyperparameters.
BATCH_SIZE = 256
EPOCHS = 1
LEARNING_RATE = 1e-3


def loss_fn(
    model: nnx.Module, inputs: jax.Array, labels_onehot: jax.Array
) -> jax.Array:
    logits = model(inputs)
    return optax.softmax_cross_entropy(logits, labels_onehot).mean()


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    inputs: jax.Array,
    labels_onehot: jax.Array,
) -> jax.Array:
    loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, labels_onehot)
    optimizer.update(model, grads)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps (overrides epochs)./"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup steps excluded from timing (compile + reach steady state).",
    )
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    # Load data.
    print("Loading CIFAR-10...")
    images, labels = load_cifar10()
    num_samples = len(images)
    print(f"Loaded {num_samples:,} training samples")

    # Create model and optimizer.
    model = ResNet18(num_classes=10, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

    # Precompute batches on device.
    num_batches = num_samples // BATCH_SIZE
    print(f"Preparing {num_batches} batches on device...")
    # Use .copy() to ensure contiguous memory layout (required for MPS backend)
    batched_images = jnp.array(
        images[: num_batches * BATCH_SIZE]
        .reshape(num_batches, BATCH_SIZE, 32, 32, 3)
        .copy()
    )
    batched_labels = jax.nn.one_hot(
        labels[: num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE).copy(), 10
    )

    # Training loop.
    num_steps = args.steps if args.steps else EPOCHS * num_batches
    warmup = min(args.warmup, max(num_steps - 1, 0))

    # We deliberately do NOT call .item() inside the hot loop. A per-step host
    # read forces a CPU<->GPU sync every iteration, serializing dispatch with
    # compute and hiding any benefit of JAX_MPS_ASYNC_DISPATCH. Instead we keep
    # each step's loss as a device array, let JAX queue the next dispatch while
    # the GPU works, and sync exactly once at the window boundary
    # (block_until_ready). Wall time over the measured window / step count is
    # the true steady-state throughput, with CPU/GPU overlap included.
    print(f"Starting training for {num_steps} steps ({warmup} warmup) ...")
    steps = tqdm(range(num_steps))
    losses = []
    measure_start = None
    measured_steps = 0
    for i in steps:
        if i == warmup:
            # Drain warmup dispatches, then start the clock on a quiet GPU.
            jax.block_until_ready(losses)
            losses.clear()
            measure_start = perf_counter()
        # Cycle through the precomputed batches so the example actually trains
        # over the dataset rather than repeating a single batch.
        batch_idx = i % num_batches
        loss = train_step(
            model, optimizer, batched_images[batch_idx], batched_labels[batch_idx]
        )
        losses.append(loss)
        if i >= warmup:
            measured_steps += 1

    # Single sync point for the whole measured window.
    jax.block_until_ready(losses)
    elapsed = perf_counter() - measure_start if measure_start is not None else 0.0
    final_loss = float(losses[-1]) if losses else float("nan")

    print(f"Final training loss: {final_loss:.3f}")
    if measured_steps:
        per_step = elapsed / measured_steps
        print(
            f"Time per step (steady state, {measured_steps} steps): {per_step:.4f} s "
            f"({measured_steps / elapsed:.1f} steps/s)"
        )


if __name__ == "__main__":
    main()
