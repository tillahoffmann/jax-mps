import jax
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
                lambda key: (
                    random.randint(random.split(key)[0], (), 0, 4),
                    random.randint(random.split(key)[1], (), 0, 5),
                ),
            ),
            OperationTestConfig(
                lambda x, idx, y: x[idx],
                lambda key: random.normal(key, (4, 5)),
                lambda key: (
                    random.randint(random.split(key)[0], (), 0, 4),
                    random.randint(random.split(key)[1], (), 0, 5),
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
            # Multi-index point gather: x[arange(n), arange(n)] extracts diagonal
            # (grad excluded: corresponding scatter pattern not yet supported)
            OperationTestConfig(
                lambda x: x[jnp.arange(4), jnp.arange(4)],
                lambda key: random.normal(key, (4, 4)),
                name="multi_index_point_gather",
                differentiable_argnums=(),
            ),
            # Batched gather via vmap (grad excluded: batched scatter not yet supported)
            OperationTestConfig(
                lambda x, idx: jax.vmap(
                    lambda xi, ii: lax.dynamic_index_in_dim(xi, ii, 0, False)
                )(x, idx),
                lambda key: random.normal(key, (3, 10)),
                lambda key: random.randint(key, (3,), 0, 10),
                name="batched_single_axis_gather",
                differentiable_argnums=(),
            ),
            # Large integer gather tests: verify integers > 2^24 are preserved
            # These test the bitcast workaround for MPS gather operations
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.array([16777217, 2**30, 2**31 - 1], dtype=numpy.uint32),
                numpy.int32(1),
                name="large_uint32_gather",
            ),
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.array([16777217, 2**30, 2**31 - 1], dtype=numpy.int32),
                numpy.int32(0),
                name="large_int32_gather",
            ),
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.array([2**40, 2**50, 2**62], dtype=numpy.uint64),
                numpy.int32(1),
                name="large_uint64_gather",
            ),
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.array([2**40, 2**50, 2**62], dtype=numpy.int64),
                numpy.int32(2),
                name="large_int64_gather",
            ),
            # Large integer scatter tests: verify integers > 2^24 are preserved in scatter
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.zeros((5,), dtype=numpy.uint32),
                numpy.int32(2),
                numpy.uint32(16777217),
                name="large_uint32_scatter",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.zeros((5,), dtype=numpy.int32),
                numpy.int32(1),
                numpy.int32(2**30),
                name="large_int32_scatter",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.zeros((5,), dtype=numpy.uint64),
                numpy.int32(3),
                numpy.uint64(2**50),
                name="large_uint64_scatter",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.zeros((5,), dtype=numpy.int64),
                numpy.int32(0),
                numpy.int64(2**62),
                name="large_int64_scatter",
            ),
        ]
