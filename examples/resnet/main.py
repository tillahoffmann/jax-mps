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
    batched_images = jnp.array(
        images[: num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE, 32, 32, 3)
    )
    batched_labels = jax.nn.one_hot(
        labels[: num_batches * BATCH_SIZE].reshape(num_batches, BATCH_SIZE), 10
    )

    # Training loop.
    num_steps = args.steps if args.steps else EPOCHS * num_batches
    batch_idx = 0
    times_per_step = []

    print(f"Starting training for {num_steps} steps ...")
    steps = tqdm(range(num_steps))
    loss = jnp.nan
    for _ in steps:
        start = perf_counter()
        loss = train_step(
            model, optimizer, batched_images[batch_idx], batched_labels[batch_idx]
        ).item()
        end = perf_counter()
        times_per_step.append(end - start)
        steps.set_description(f"loss = {loss:.3f}")

    print(f"Final training loss: {loss:.3f}")
    times_per_step = times_per_step[len(times_per_step) // 2 :]
    print(
        f"Time per step (second half): {sum(times_per_step) / len(times_per_step):.3f}"
    )


if __name__ == "__main__":
    main()
