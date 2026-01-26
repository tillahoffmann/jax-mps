import jax
from flax import nnx
from jax import numpy as jnp


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
