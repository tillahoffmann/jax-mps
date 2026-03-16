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
            # dynamic_update_slice with dynamic start indices (issue #87)
            OperationTestConfig(
                lambda x, update, idx: lax.dynamic_update_slice(
                    x, update, (idx, jnp.int32(0))
                ),
                lambda key: random.normal(key, (5, 3)),
                lambda key: random.normal(key, (2, 3)),
                lambda key: random.randint(key, (), 0, 4, dtype=jnp.int32),
                name="dynamic_update_slice_dynamic_idx",
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
            OperationTestConfig(
                lambda x: x[jnp.arange(4), jnp.arange(4)],
                lambda key: random.normal(key, (4, 4)),
                name="multi_index_point_gather",
            ),
            # Batched gather via vmap
            OperationTestConfig(
                lambda x, idx: jax.vmap(
                    lambda xi, ii: lax.dynamic_index_in_dim(xi, ii, 0, False)
                )(x, idx),
                lambda key: random.normal(key, (3, 10)),
                lambda key: random.randint(key, (3,), 0, 10),
                name="batched_single_axis_gather",
            ),
            # Batched gather via vmap with no collapsed dims (issue #74)
            OperationTestConfig(
                lambda x, idx: jax.vmap(lambda state, i: state[i], in_axes=(0, 0))(
                    x, idx
                ),
                lambda key: random.normal(key, (4, 5)),
                lambda key: random.randint(key, (4,), 0, 5),
                name="batched_gather_no_collapse",
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
            # Multi-dim index scatter: scatter values along diagonal of a matrix
            # Exercises general ScatterND with multi-point, multi-dim index vectors
            # indices shape [N, K] where N=4 scatter points, K=2 index dims
            OperationTestConfig(
                lambda x, vals: x.at[numpy.arange(4), numpy.arange(4)].add(vals),
                lambda key: jnp.zeros((4, 4), dtype=jnp.float32),
                lambda key: random.normal(key, (4,)),
                differentiable_argnums=(0,),
                name="scatter_multi_dim_diagonal_add",
            ),
            # Grad of multi-dim scatter with respect to updates
            OperationTestConfig(
                lambda x, vals: x.at[numpy.arange(4), numpy.arange(4)].add(vals),
                lambda key: jnp.zeros((4, 4), dtype=jnp.float32),
                lambda key: random.normal(key, (4,)),
                differentiable_argnums=(1,),
                name="scatter_multi_dim_diagonal_add_grad_updates",
            ),
            # Batched scatter using vmap - tests numStableHLOBatch > 0
            # These crash due to incorrect handling of StableHLO batch dimensions
            # in the general scatter fallback (reshape loses batch dims). See PR #49.
            OperationTestConfig(
                lambda x, idx, val: jax.vmap(lambda a, i, v: a.at[i].set(v))(
                    x, idx, val
                ),
                lambda key: random.normal(key, (3, 5)),
                lambda key: random.randint(key, (3,), 0, 5),
                lambda key: random.normal(key, (3,)),
                differentiable_argnums=(0, 2),
                name="scatter_vmap_simple",
            ),
            OperationTestConfig(
                lambda x, idx, val: jax.vmap(lambda a, i, v: a.at[i].add(v))(
                    x, idx, val
                ),
                lambda key: jnp.zeros((3, 5), dtype=jnp.float32),
                lambda key: jnp.array([[0, 2], [1, 3], [2, 4]]),
                lambda key: random.normal(key, (3, 2)),
                differentiable_argnums=(0, 2),
                name="scatter_vmap_multi_point",
            ),
            OperationTestConfig(
                lambda x, vals: jax.vmap(
                    lambda a, v: a.at[numpy.arange(2), numpy.arange(2)].add(v)
                )(x, vals),
                lambda key: jnp.zeros((3, 4, 4), dtype=jnp.float32),
                lambda key: random.normal(key, (3, 2)),
                differentiable_argnums=(0,),
                name="scatter_vmap_2d_diagonal",
            ),
            # Window scatter: slice update with no inserted window dims (issue #89)
            # jnp.zeros((2,)).at[1:].set(-1.0) produces scatter with
            # scatterDimsToOperandDims.size=1, insertedWindowDims.size=0
            OperationTestConfig(
                lambda x: x.at[1:].set(-1.0),
                lambda key: jnp.zeros((2,), dtype=jnp.float32),
                name="window_scatter_slice_set_1d",
            ),
            OperationTestConfig(
                lambda x: x.at[1:4].set(jnp.array([10.0, 20.0, 30.0])),
                lambda key: jnp.zeros((5,), dtype=jnp.float32),
                name="window_scatter_slice_set_1d_multi",
            ),
            OperationTestConfig(
                lambda x: x.at[1:3].set(jnp.ones((2, 4))),
                lambda key: jnp.zeros((5, 4), dtype=jnp.float32),
                name="window_scatter_slice_set_2d",
            ),
            # Partial-index gather with non-sorted startIndexMap.
            # start_index_map=(2, 0) means idx[0]->dim2, idx[1]->dim0.
            # Strides must follow sorted collapsed-dim order, not startIndexMap order.
            # Indices [3, 2] select dim2=3, dim0=2 => x[2, :, 3].
            # With shape (3,4,5), wrong strides give linear idx 11 instead of 13.
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array([[3, 2]]),
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(1,),
                        collapsed_slice_dims=(0, 2),
                        start_index_map=(2, 0),
                    ),
                    slice_sizes=(1, 4, 1),
                ),
                lambda key: random.normal(key, (3, 4, 5)),
                name="partial_gather_reversed_index_map",
            ),
            # Fix 3: Partial-index gather with scalar indices.
            # indices shape [2] with indexVectorDim=0 => each sub-index is scalar.
            # linearIdx must be padded to match flatOperand.ndim() for take_along_axis.
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array([1, 0]),
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(0,),
                        collapsed_slice_dims=(0, 2),
                        start_index_map=(0, 2),
                    ),
                    slice_sizes=(1, 4, 1),
                ),
                lambda key: random.normal(key, (3, 4, 5)),
                name="partial_gather_scalar_indices",
            ),
            # Multi-dim gather with no collapsed dims (dynamic sub-tensor slice).
            # start_index_map=[1,2] with empty collapsed_slice_dims: extracts a
            # [2,1,1] sub-tensor from [2,3,3] at dynamic (row, col) start position.
            # Used by batched LU pivoting in jnp.linalg.inv.
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array([[1, 2]]),
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(0, 1, 2),
                        collapsed_slice_dims=(),
                        start_index_map=(1, 2),
                    ),
                    slice_sizes=(2, 1, 1),
                ),
                lambda key: random.normal(key, (2, 3, 3)),
                name="multi_dim_gather_no_collapse",
            ),
            # Batched gather with offset dims: per-batch column permutation.
            # operand_batching_dims=[0], start_index_map=[1] with offset_dims
            # that require proper index expansion (not just trailing 1s).
            # Used by batched LU solve in jnp.linalg.inv.
            OperationTestConfig(
                lambda x, idx: jax.vmap(lambda xi, ii: xi[:, ii])(x, idx),
                lambda key: random.normal(key, (2, 3, 4)),
                lambda key: random.randint(key, (2, 3), 0, 4),
                name="batched_gather_offset_dims",
            ),
        ]
