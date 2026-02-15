import numpy
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig


def make_slice_op_configs():
    with OperationTestConfig.module_name("slice"):
        return [
            OperationTestConfig(
                lambda x, idx: x[idx],
                lambda rng: rng.normal(size=(4, 5)),
                lambda rng: (rng.integers(4), rng.integers(5)),
            ),
            OperationTestConfig(
                lambda x, idx, y: x[idx],
                lambda rng: rng.normal(size=(4, 5)),
                lambda rng: (rng.integers(4), rng.integers(5)),
                lambda rng: numpy.asarray(7.0),
            ),
            OperationTestConfig(
                lambda x: lax.dynamic_slice(x, (2,), (4,)),
                lambda rng: rng.normal(size=(10,)),
            ),
            OperationTestConfig(
                lambda x, idx: jnp.take(x, idx, axis=0),
                lambda rng: rng.normal(size=(5, 3)),
                numpy.array([0, 2, 4]),
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda rng: rng.normal(size=(5, 3)),
                numpy.array([0, 2]),
                lambda rng: rng.normal(size=(2, 3)),
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda rng: jnp.zeros((10, 1, 4), dtype=jnp.bfloat16),
                lambda rng: numpy.int32(0),
                lambda rng: jnp.ones((1, 4), dtype=jnp.bfloat16),
                name="scalar_index_set_rank_squeezed_update",
            ),
            OperationTestConfig(
                lambda x, val: x.at[0, 0, 0].set(val),
                lambda rng: jnp.zeros((2, 2, 2), dtype=jnp.float32),
                lambda rng: jnp.array(3.14, dtype=jnp.float32),
                name="scalar_update_rank_mismatch_gt_1",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda rng: numpy.int32(0),
                lambda rng: jnp.array(5.0, dtype=jnp.float32),
                name="slice_update_scalar_broadcast_rank3",
            ),
            OperationTestConfig(
                lambda x: x.at[0].set(1.0),
                lambda rng: rng.normal(size=(10,)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].add(1.0),
                lambda rng: rng.normal(size=(10,)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].divide(2.0),
                lambda rng: rng.normal(size=(10,)),
            ),
            OperationTestConfig(
                lambda x, update: lax.dynamic_update_slice(x, update, (1, 0)),
                lambda rng: rng.normal(size=(5, 3)),
                lambda rng: rng.normal(size=(2, 3)),
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
                lambda rng: rng.normal(size=(10, 4)).astype(numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                lambda rng: rng.normal(size=(3, 4)).astype(numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].max(updates),
                lambda rng: rng.normal(size=(10, 4)).astype(numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                lambda rng: rng.normal(size=(3, 4)).astype(numpy.float32),
            ),
        ]
