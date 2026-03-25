"""Test configurations for matmul and dot_general operations.

These tests focus on batched matrix-matrix multiplication cases that exercise
the dot_general batch dimension handling.
"""

import numpy
from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def make_matmul_op_configs():
    """Generate test configurations for matrix multiplication operations."""

    with OperationTestConfig.module_name("matmul"):
        # Basic 2D matmul (M, K) @ (K, N) -> (M, N)
        yield OperationTestConfig(
            jnp.matmul,
            lambda key: random.normal(key, (4, 5)),
            lambda key: random.normal(key, (5, 3)),
            name="2d_basic",
        )

        # Single batch dimension: (B, M, K) @ (B, K, N) -> (B, M, N)
        yield OperationTestConfig(
            jnp.matmul,
            lambda key: random.normal(key, (2, 4, 5)),
            lambda key: random.normal(key, (2, 5, 3)),
            name="batched_1d",
        )

        # Multiple batch dimensions: (B1, B2, M, K) @ (B1, B2, K, N) -> (B1, B2, M, N)
        yield OperationTestConfig(
            jnp.matmul,
            lambda key: random.normal(key, (2, 3, 4, 5)),
            lambda key: random.normal(key, (2, 3, 5, 6)),
            name="batched_2d",
        )

        # Three batch dimensions
        yield OperationTestConfig(
            jnp.matmul,
            lambda key: random.normal(key, (2, 2, 2, 3, 4)),
            lambda key: random.normal(key, (2, 2, 2, 4, 5)),
            name="batched_3d",
        )

        # Test with einsum for more explicit control
        yield OperationTestConfig(
            lambda x, y: jnp.einsum("bij,bjk->bik", x, y),
            lambda key: random.normal(key, (3, 4, 5)),
            lambda key: random.normal(key, (3, 5, 6)),
            name="einsum_batched",
        )

        # Einsum with two batch dimensions
        yield OperationTestConfig(
            lambda x, y: jnp.einsum("abij,abjk->abik", x, y),
            lambda key: random.normal(key, (2, 3, 4, 5)),
            lambda key: random.normal(key, (2, 3, 5, 6)),
            name="einsum_batched_2d",
        )

        # lax.dot_general with explicit dimension numbers
        # Standard batched matmul: contract last of lhs with second-to-last of rhs
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),  # (contract, batch)
            ),
            lambda key: random.normal(key, (3, 4, 5)),
            lambda key: random.normal(key, (3, 5, 6)),
            name="dot_general_batched",
        )

        # lax.dot_general with two batch dimensions
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
            ),
            lambda key: random.normal(key, (2, 3, 4, 5)),
            lambda key: random.normal(key, (2, 3, 5, 6)),
            name="dot_general_batched_2d",
        )

        # lax.dot_general: contract on first dimension of lhs (transposed lhs)
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
            ),
            lambda key: random.normal(key, (3, 5, 4)),  # (B, K, M)
            lambda key: random.normal(key, (3, 5, 6)),  # (B, K, N)
            name="dot_general_lhs_transposed",
        )

        # lax.dot_general: contract on last dimension of rhs (transposed rhs)
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            ),
            lambda key: random.normal(key, (3, 4, 5)),  # (B, M, K)
            lambda key: random.normal(key, (3, 6, 5)),  # (B, N, K)
            name="dot_general_rhs_transposed",
        )

        # lax.dot_general: both transposed
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((1,), (2,)), ((0,), (0,))),
            ),
            lambda key: random.normal(key, (3, 5, 4)),  # (B, K, M)
            lambda key: random.normal(key, (3, 6, 5)),  # (B, N, K)
            name="dot_general_both_transposed",
        )

        # Edge case: multiple free dimensions (not matmul-like)
        # LHS: (B, M1, M2, K), RHS: (B, K, N1, N2) -> contracts K, 2 free dims each
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(
                    ((3,), (1,)),
                    ((0,), (0,)),
                ),  # contract dim 3/1, batch dim 0
            ),
            lambda key: random.normal(key, (2, 3, 4, 5)),
            lambda key: random.normal(key, (2, 5, 6, 7)),
            name="dot_general_multiple_free_dims",
        )

        # Matrix-vector: (M, K) @ (K,) -> (M,)
        yield OperationTestConfig(
            jnp.matmul,
            lambda key: random.normal(key, (4, 5)),
            lambda key: random.normal(key, (5,)),
            name="matvec",
        )

        # Vector-matrix: (K,) @ (K, N) -> (N,)
        yield OperationTestConfig(
            jnp.matmul,
            lambda key: random.normal(key, (5,)),
            lambda key: random.normal(key, (5, 3)),
            name="vecmat",
        )

        # Vector-vector (dot product): (K,) @ (K,) -> ()
        yield OperationTestConfig(
            jnp.dot,
            lambda key: random.normal(key, (5,)),
            lambda key: random.normal(key, (5,)),
            name="vecdot",
        )

        # Exact example from issue #53: jnp.dot(ones((3,4)), ones(4))
        yield OperationTestConfig(
            jnp.dot,
            lambda key: jnp.ones((3, 4)),
            lambda key: jnp.ones((4,)),
            name="issue_53_matvec",
        )

        # Outer product: (M,) x (N,) -> (M, N) with no contracting dimensions
        # This pattern is generated by gradient computation of matrix-vector products
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(x, y, dimension_numbers=(([], []), ([], []))),
            lambda key: random.normal(key, (3,)),
            lambda key: random.normal(key, (4,)),
            name="outer_product",
        )

        # Edge case: zero batch size (empty batch dimension)
        # CPU handles this correctly, returning empty arrays with the right shape.
        yield OperationTestConfig(
            jnp.matmul,
            numpy.zeros((0, 3, 4), dtype=numpy.float32),
            numpy.zeros((0, 4, 5), dtype=numpy.float32),
            name="batched_zero_batch",
        )

        # Integer dot products (MLX matmul only supports float, so handler must cast)
        yield OperationTestConfig(
            jnp.dot,
            numpy.array([1, 2, 3], dtype=numpy.int32),
            numpy.array([4, 5, 6], dtype=numpy.int32),
            differentiable_argnums=(),
            name="dot_int32",
        )
        yield OperationTestConfig(
            jnp.matmul,
            numpy.array([[1, 2], [3, 4]], dtype=numpy.int32),
            numpy.array([[5, 6], [7, 8]], dtype=numpy.int32),
            differentiable_argnums=(),
            name="matmul_int32",
        )
        yield OperationTestConfig(
            jnp.dot,
            numpy.array([True, False, True], dtype=numpy.bool_),
            numpy.array([True, True, False], dtype=numpy.bool_),
            differentiable_argnums=(),
            name="dot_bool",
        )
        yield OperationTestConfig(
            jnp.matmul,
            numpy.array([[1, 2], [3, 4]], dtype=numpy.uint32),
            numpy.array([[5, 6], [7, 8]], dtype=numpy.uint32),
            differentiable_argnums=(),
            name="matmul_uint32",
        )
