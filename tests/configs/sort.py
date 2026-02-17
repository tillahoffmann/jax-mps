from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def make_sort_op_configs():
    """Test configs for sort, topk, argmax/argmin operations."""
    with OperationTestConfig.module_name("sort"):
        configs = []

        # =============================================================
        # jnp.sort - single tensor sort
        # Gradients enabled - uses scatter with batching dims
        # =============================================================

        # 1D sort
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x),
                lambda key: random.normal(key, (16,)),
                name="jnp.sort.1d",
            )
        )

        # 2D sort - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=0),
                lambda key: random.normal(key, (4, 8)),
                name="jnp.sort.2d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=1),
                lambda key: random.normal(key, (4, 8)),
                name="jnp.sort.2d.axis1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=-1),
                lambda key: random.normal(key, (4, 8)),
                name="jnp.sort.2d.axis_neg1",
            )
        )

        # 3D sort - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=0),
                lambda key: random.normal(key, (2, 4, 8)),
                name="jnp.sort.3d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=1),
                lambda key: random.normal(key, (2, 4, 8)),
                name="jnp.sort.3d.axis1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=2),
                lambda key: random.normal(key, (2, 4, 8)),
                name="jnp.sort.3d.axis2",
            )
        )

        # Descending sort
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=-1, descending=True),
                lambda key: random.normal(key, (4, 8)),
                name="jnp.sort.descending",
            )
        )

        # =============================================================
        # lax.sort_key_val - sort with associated values
        # Gradients enabled for both keys and values
        # =============================================================

        # 2D key-value sort on axis 0
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=0),
                lambda key: random.normal(key, (4, 8)),
                lambda key: random.normal(key, (4, 8)),
                name="lax.sort_key_val.2d.axis0",
            )
        )

        # 2D key-value sort on axis 1
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=1),
                lambda key: random.normal(key, (4, 8)),
                lambda key: random.normal(key, (4, 8)),
                name="lax.sort_key_val.2d.axis1",
            )
        )

        # 3D key-value sort
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=0),
                lambda key: random.normal(key, (2, 4, 8)),
                lambda key: random.normal(key, (2, 4, 8)),
                name="lax.sort_key_val.3d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=2),
                lambda key: random.normal(key, (2, 4, 8)),
                lambda key: random.normal(key, (2, 4, 8)),
                name="lax.sort_key_val.3d.axis2",
            )
        )

        # Key-value with stable sort
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(
                    keys, vals, dimension=1, is_stable=True
                ),
                lambda key: random.normal(key, (4, 8)),
                lambda key: jnp.arange(32).reshape(4, 8).astype(jnp.float32),
                name="lax.sort_key_val.stable",
            )
        )

        # =============================================================
        # lax.top_k - top k elements and indices
        # Returns (values, indices) - values are differentiable
        # =============================================================

        # 1D top_k
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                lambda key: random.normal(key, (16,)),
                name="lax.top_k.1d",
            )
        )

        # 2D top_k - different k values
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 1),
                lambda key: random.normal(key, (4, 8)),
                name="lax.top_k.2d.k1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                lambda key: random.normal(key, (4, 8)),
                name="lax.top_k.2d.k3",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 5),
                lambda key: random.normal(key, (4, 8)),
                name="lax.top_k.2d.k5",
            )
        )

        # 3D top_k
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 2),
                lambda key: random.normal(key, (2, 4, 8)),
                name="lax.top_k.3d",
            )
        )

        # 4D top_k
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                lambda key: random.normal(key, (2, 2, 4, 8)),
                name="lax.top_k.4d",
            )
        )

        # top_k after transpose (tests non-last-axis behavior)
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(jnp.swapaxes(x, 0, 2), 2),
                lambda key: random.normal(key, (2, 4, 8)),
                name="lax.top_k.transposed",
            )
        )

        # top_k with k equal to axis size
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 5),
                lambda key: random.normal(key, (4, 8)),
                name="lax.top_k.k_equals_size",
            )
        )

        # =============================================================
        # jnp.argmax - argmax (multi-result reduce with max)
        # =============================================================

        # 1D argmax
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x),
                lambda key: random.normal(key, (16,)),
                differentiable_argnums=(),
                name="jnp.argmax.1d",
            )
        )

        # 2D argmax - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=0),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.2d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=1),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.2d.axis1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=-1),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.2d.axis_neg1",
            )
        )

        # 3D argmax
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=0),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.3d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=1),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.3d.axis1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=2),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.3d.axis2",
            )
        )

        # argmax with keepdims
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=1, keepdims=True),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.keepdims",
            )
        )

        # =============================================================
        # jnp.argmin - argmin (multi-result reduce with min)
        # =============================================================

        # 1D argmin
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x),
                lambda key: random.normal(key, (16,)),
                differentiable_argnums=(),
                name="jnp.argmin.1d",
            )
        )

        # 2D argmin - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=0),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.2d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=1),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.2d.axis1",
            )
        )

        # 3D argmin
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=0),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.3d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=2),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.3d.axis2",
            )
        )

        # argmin with keepdims
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=0, keepdims=True),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.keepdims",
            )
        )

        # =============================================================
        # jnp.argsort - indices that would sort
        # =============================================================

        # 1D argsort
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x),
                lambda key: random.normal(key, (16,)),
                differentiable_argnums=(),
                name="jnp.argsort.1d",
            )
        )

        # 2D argsort - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x, axis=0),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argsort.2d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x, axis=1),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argsort.2d.axis1",
            )
        )

        # argsort descending
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x, descending=True),
                lambda key: random.normal(key, (16,)),
                differentiable_argnums=(),
                name="jnp.argsort.descending",
            )
        )

        return configs
