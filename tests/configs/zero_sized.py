"""Edge case tests for zero-sized tensor support.

Tests operations involving tensors where at least one dimension is 0.
"""

import numpy
from jax import numpy as jnp

from .util import OperationTestConfig


def make_zero_sized_op_configs():
    with OperationTestConfig.module_name("zero_sized"):
        # Basic zero-sized shapes
        yield OperationTestConfig(
            lambda x: x + 1,
            numpy.zeros((0,), dtype=numpy.float32),
            name="unary_empty_1d",
        )
        yield OperationTestConfig(
            lambda x: x + 1,
            numpy.zeros((3, 0), dtype=numpy.float32),
            name="unary_zero_middle_dim",
        )
        yield OperationTestConfig(
            lambda x: x + 1,
            numpy.zeros((3, 4, 0), dtype=numpy.float32),
            name="unary_zero_last_dim",
        )

        # Broadcasting with zero-sized arrays
        # Note: Gradient computation hangs on MPS, so disable gradients
        yield OperationTestConfig(
            lambda x, y: x + y,
            numpy.zeros((0, 3), dtype=numpy.float32),
            numpy.ones((1, 3), dtype=numpy.float32),
            differentiable_argnums=(),
            name="broadcast_empty_with_nonempty",
        )

        # Concatenation with empty and non-empty arrays
        # Bug: MPS returns zeros instead of the non-empty array's values
        yield OperationTestConfig(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            numpy.zeros((0, 3), dtype=numpy.float32),
            numpy.ones((2, 3), dtype=numpy.float32),
            differentiable_argnums=(),
            name="concat_empty_nonempty",
        )
        yield OperationTestConfig(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            numpy.ones((2, 3), dtype=numpy.float32),
            numpy.zeros((0, 3), dtype=numpy.float32),
            differentiable_argnums=(),
            name="concat_nonempty_empty",
        )

        # Reduction over zero-sized axis producing non-zero output
        yield OperationTestConfig(
            lambda x: jnp.sum(x, axis=0),
            numpy.zeros((0, 3), dtype=numpy.float32),
            name="sum_over_zero_axis",
        )
        yield OperationTestConfig(
            lambda x: jnp.sum(x, axis=1),
            numpy.zeros((3, 0), dtype=numpy.float32),
            name="sum_over_zero_axis_last",
        )

        # Product reduction (empty product = 1, multiplicative identity)
        # Bug: MPS returns zeros instead of ones
        yield OperationTestConfig(
            lambda x: jnp.prod(x, axis=0),
            numpy.zeros((0, 3), dtype=numpy.float32),
            differentiable_argnums=(),
            name="prod_over_zero_axis",
        )

        # Mean over empty should produce NaN (0/0)
        # Bug: MPS returns 0 instead of NaN
        yield OperationTestConfig(
            lambda x: jnp.mean(x),
            numpy.zeros((0,), dtype=numpy.float32),
            differentiable_argnums=(),
            name="mean_empty_scalar",
        )

        # Take with empty indices
        yield OperationTestConfig(
            lambda x, indices: jnp.take(x, indices),
            numpy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=numpy.float32),
            numpy.array([], dtype=numpy.int32),
            differentiable_argnums=(0,),
            name="take_empty_indices",
        )

        # Matmul with zero-sized contracting dimension
        # Result should be zeros (sum over no elements)
        yield OperationTestConfig(
            jnp.matmul,
            numpy.zeros((3, 0), dtype=numpy.float32),
            numpy.zeros((0, 4), dtype=numpy.float32),
            name="matmul_zero_contracting",
        )

        # Stack empty arrays
        yield OperationTestConfig(
            lambda x, y: jnp.stack([x, y]),
            numpy.zeros((0,), dtype=numpy.float32),
            numpy.zeros((0,), dtype=numpy.float32),
            name="stack_empty_arrays",
        )

        # Reshape to zero-sized
        yield OperationTestConfig(
            lambda x: jnp.reshape(x, (0, 5)),
            numpy.zeros((0,), dtype=numpy.float32),
            name="reshape_to_zero",
        )

        # Transpose empty
        yield OperationTestConfig(
            jnp.transpose,
            numpy.zeros((0, 3, 4), dtype=numpy.float32),
            name="transpose_empty",
        )

        # Where with empty condition
        yield OperationTestConfig(
            lambda cond, x, y: jnp.where(cond, x, y),
            numpy.array([], dtype=numpy.bool_),
            numpy.array([], dtype=numpy.float32),
            numpy.array([], dtype=numpy.float32),
            differentiable_argnums=(1, 2),
            name="where_empty_condition",
        )

        # Slicing to empty result
        yield OperationTestConfig(
            lambda x: x[2:2],
            numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32),
            name="slice_to_empty",
        )

        # All/any on empty
        # all([]) = True (vacuous truth), any([]) = False
        # Bug: MPS returns False for all (should be True)
        yield OperationTestConfig(
            jnp.all,
            numpy.array([], dtype=numpy.bool_),
            differentiable_argnums=(),
            name="all_empty",
        )
        yield OperationTestConfig(
            jnp.any,
            numpy.array([], dtype=numpy.bool_),
            differentiable_argnums=(),
            name="any_empty",
        )

        # Dot product with one empty operand
        yield OperationTestConfig(
            jnp.dot,
            numpy.zeros((0,), dtype=numpy.float32),
            numpy.zeros((0,), dtype=numpy.float32),
            name="dot_empty_vectors",
        )
