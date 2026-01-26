"""
ResNet18 training on CIFAR-10 using Flax NNX.
"""

import argparse
import pickle
import tarfile
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Hyperparameters.
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
CACHE_DIR = Path.home() / ".cache" / "cifar10"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def load_cifar10() -> tuple[np.ndarray, np.ndarray]:
    """Download and load CIFAR-10 training data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tarball_path = CACHE_DIR / "cifar-10-python.tar.gz"

    # Download if not cached.
    if not tarball_path.exists():
        print(f"Downloading CIFAR-10 to {tarball_path}...")
        urllib.request.urlretrieve(CIFAR10_URL, tarball_path)

    # Extract and load training batches.
    images, labels = [], []
    with tarfile.open(tarball_path, "r:gz") as tar:
        for i in range(1, 6):
            member = tar.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            assert member is not None
            batch = pickle.load(member, encoding="bytes")
            images.append(batch[b"data"])
            labels.append(batch[b"labels"])

    # Reshape to (N, 32, 32, 3) and normalize to [0, 1].
    images = np.concatenate(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    images = images.astype(np.float32) / 255.0
    labels = np.concatenate(labels).astype(np.int32)

    # Normalize with mean/std.
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))
    images = (images - mean) / std

    return images, labels


class ResNetBlock(nnx.Module):
    """Basic ResNet block with two 3x3 convolutions."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        strides: tuple[int, int] = (1, 1),
        rngs: nnx.Rngs,
    ):
        self.conv1 = nnx.Conv(
            in_features,
            out_features,
            kernel_size=(3, 3),
            strides=strides,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(out_features, momentum=0.9, epsilon=1e-5, rngs=rngs)

        self.conv2 = nnx.Conv(
            out_features,
            out_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(out_features, momentum=0.9, epsilon=1e-5, rngs=rngs)
        # Initialize bn2 scale to zero for better training dynamics
        assert self.bn2.scale is not None
        self.bn2.scale[...] = jnp.zeros_like(self.bn2.scale[...])

        # Projection shortcut if dimensions change
        self.needs_projection = in_features != out_features or strides != (1, 1)
        if self.needs_projection:
            self.conv_proj = nnx.Conv(
                in_features,
                out_features,
                kernel_size=(1, 1),
                strides=strides,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            )
            self.bn_proj = nnx.BatchNorm(
                out_features, momentum=0.9, epsilon=1e-5, rngs=rngs
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = nnx.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.needs_projection:
            residual = self.conv_proj(residual)
            residual = self.bn_proj(residual)

        return nnx.relu(residual + y)


class ResNet18(nnx.Module):
    """ResNet18 for CIFAR-10 (32x32 images)."""

    def __init__(
        self,
        num_classes: int = 10,
        num_filters: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_filters = num_filters

        # Initial convolution - use 3x3 kernel for CIFAR's small images
        self.conv_init = nnx.Conv(
            3,  # RGB input
            num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn_init = nnx.BatchNorm(num_filters, momentum=0.9, epsilon=1e-5, rngs=rngs)

        # ResNet18 stage sizes: [2, 2, 2, 2]
        # Stage 1: 64 filters
        self.stage1_block1 = ResNetBlock(
            num_filters, num_filters, strides=(1, 1), rngs=rngs
        )
        self.stage1_block2 = ResNetBlock(
            num_filters, num_filters, strides=(1, 1), rngs=rngs
        )

        # Stage 2: 128 filters, first block has stride 2
        self.stage2_block1 = ResNetBlock(
            num_filters, num_filters * 2, strides=(2, 2), rngs=rngs
        )
        self.stage2_block2 = ResNetBlock(
            num_filters * 2, num_filters * 2, strides=(1, 1), rngs=rngs
        )

        # Stage 3: 256 filters, first block has stride 2
        self.stage3_block1 = ResNetBlock(
            num_filters * 2, num_filters * 4, strides=(2, 2), rngs=rngs
        )
        self.stage3_block2 = ResNetBlock(
            num_filters * 4, num_filters * 4, strides=(1, 1), rngs=rngs
        )

        # Stage 4: 512 filters, first block has stride 2
        self.stage4_block1 = ResNetBlock(
            num_filters * 4, num_filters * 8, strides=(2, 2), rngs=rngs
        )
        self.stage4_block2 = ResNetBlock(
            num_filters * 8, num_filters * 8, strides=(1, 1), rngs=rngs
        )

        # Classification head
        self.dense = nnx.Linear(num_filters * 8, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Initial conv block (no max pool for CIFAR)
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = nnx.relu(x)

        # Stage 1
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)

        # Stage 2
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)

        # Stage 3
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)

        # Stage 4
        x = self.stage4_block1(x)
        x = self.stage4_block2(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Classification
        x = self.dense(x)

        return x


def main():
    """Train ResNet18 on CIFAR-10."""
    parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR-10")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps (overrides epochs)"
    )
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    # Load data.
    print("Loading CIFAR-10...")
    images, labels = load_cifar10()
    num_samples = len(images)
    print(f"Loaded {num_samples} training samples")

    # Create model and optimizer.
    model = ResNet18(num_classes=10, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, inputs, labels_onehot):
        def loss_fn(model):
            logits = model(inputs)
            return optax.softmax_cross_entropy(logits, labels_onehot).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

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
    total_steps = args.steps if args.steps else EPOCHS * num_batches
    step = 0
    batch_idx = 0

    while step < total_steps:
        epoch_loss = 0.0
        epoch_steps = 0

        for _ in range(num_batches):
            if step >= total_steps:
                break

            loss = train_step(
                model, optimizer, batched_images[batch_idx], batched_labels[batch_idx]
            )
            batch_idx = (batch_idx + 1) % num_batches
            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1

        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            print(f"Step {step}/{total_steps}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
