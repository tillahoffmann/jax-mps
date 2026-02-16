import numpy
from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def make_slice_op_configs():
    with OperationTestConfig.module_name("slice"):
        return [
            OperationTestConfig(
                lambda x, idx: x[idx],
                lambda key: random.normal(key, (4, 5)),
                # FIXME: Should split key but MPS has a bug where random.split produces
                # different values than CPU. Using same key twice works because randint
                # with different ranges produces different values deterministically.
                lambda key: (
                    random.randint(key, (), 0, 4),
                    random.randint(key, (), 0, 5),
                ),
            ),
            OperationTestConfig(
                lambda x, idx, y: x[idx],
                lambda key: random.normal(key, (4, 5)),
                # FIXME: Same MPS random.split bug as above.
                lambda key: (
                    random.randint(key, (), 0, 4),
                    random.randint(key, (), 0, 5),
                ),
                lambda key: numpy.asarray(7.0),
            ),
            OperationTestConfig(
                lambda x: lax.dynamic_slice(x, (2,), (4,)),
                lambda key: random.normal(key, (10,)),
            ),
            OperationTestConfig(
                lambda x, idx: jnp.take(x, idx, axis=0),
                lambda key: random.normal(key, (5, 3)),
                numpy.array([0, 2, 4]),
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda key: random.normal(key, (5, 3)),
                numpy.array([0, 2]),
                lambda key: random.normal(key, (2, 3)),
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda key: jnp.zeros((10, 1, 4), dtype=jnp.float32),
                lambda key: numpy.int32(0),
                lambda key: jnp.ones((1, 4), dtype=jnp.float32),
                name="scalar_index_set_rank_squeezed_update",
            ),
            OperationTestConfig(
                lambda x, val: x.at[0, 0, 0].set(val),
                lambda key: jnp.zeros((2, 2, 2), dtype=jnp.float32),
                lambda key: jnp.array(3.14, dtype=jnp.float32),
                name="scalar_update_rank_mismatch_gt_1",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda key: jnp.zeros((2, 2, 2), dtype=jnp.float32),
                lambda key: numpy.int32(0),
                lambda key: jnp.array(5.0, dtype=jnp.float32),
                name="slice_update_scalar_broadcast_rank3",
            ),
            # Full-index gather: x[i, j, k] on rank-3 tensor returns scalar
            OperationTestConfig(
                lambda x: x[1, 2, 0],
                lambda key: random.normal(key, (3, 4, 2)),
                name="full_index_gather_rank3",
            ),
            # ScatterND with add mode (not just set)
            OperationTestConfig(
                lambda x, val: x.at[0, 0, 0].add(val),
                lambda key: jnp.ones((2, 2, 2), dtype=jnp.float32),
                lambda key: jnp.array(5.0, dtype=jnp.float32),
                name="scatternd_add_mode",
            ),
            # Higher rank tensor (rank 4)
            OperationTestConfig(
                lambda x, val: x.at[0, 0, 0, 0].set(val),
                lambda key: jnp.zeros((2, 3, 4, 5), dtype=jnp.float32),
                lambda key: jnp.array(1.0, dtype=jnp.float32),
                name="full_index_scatter_rank4",
            ),
            # Non-zero indices
            OperationTestConfig(
                lambda x, val: x.at[1, 1, 1].set(val),
                lambda key: jnp.zeros((3, 3, 3), dtype=jnp.float32),
                lambda key: jnp.array(7.0, dtype=jnp.float32),
                name="full_index_scatter_nonzero",
            ),
            # Mixed index pattern
            OperationTestConfig(
                lambda x, val: x.at[2, 0, 1].set(val),
                lambda key: jnp.zeros((4, 3, 2), dtype=jnp.float32),
                lambda key: jnp.array(9.0, dtype=jnp.float32),
                name="full_index_scatter_mixed",
            ),
            OperationTestConfig(
                lambda x: x.at[0].set(1.0),
                lambda key: random.normal(key, (10,)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].add(1.0),
                lambda key: random.normal(key, (10,)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].divide(2.0),
                lambda key: random.normal(key, (10,)),
            ),
            OperationTestConfig(
                lambda x, update: lax.dynamic_update_slice(x, update, (1, 0)),
                lambda key: random.normal(key, (5, 3)),
                lambda key: random.normal(key, (2, 3)),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].add(updates),
                numpy.zeros((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.ones((3, 4), dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].subtract(updates),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 0.1, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].mul(updates, unique_indices=True),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 2.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].divide(updates, unique_indices=True),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 2.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].power(updates, unique_indices=True),
                numpy.full((10, 4), 2.0, dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 3.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].min(updates),
                lambda key: random.normal(key, (10, 4)),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                lambda key: random.normal(key, (3, 4)),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].max(updates),
                lambda key: random.normal(key, (10, 4)),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                lambda key: random.normal(key, (3, 4)),
            ),
        ]
