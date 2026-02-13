"""Test configurations for matmul and dot_general operations.

These tests focus on batched matrix-matrix multiplication cases that exercise
the dot_general batch dimension handling.
"""

import numpy
import pytest
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig, xfail_match


def make_matmul_op_configs():
    """Generate test configurations for matrix multiplication operations."""

    with OperationTestConfig.module_name("matmul"):
        # Basic 2D matmul (M, K) @ (K, N) -> (M, N)
        yield OperationTestConfig(
            jnp.matmul,
            lambda rng: rng.standard_normal((4, 5)).astype(numpy.float32),
            lambda rng: rng.standard_normal((5, 3)).astype(numpy.float32),
            name="2d_basic",
        )

        # Single batch dimension: (B, M, K) @ (B, K, N) -> (B, M, N)
        yield OperationTestConfig(
            jnp.matmul,
            lambda rng: rng.standard_normal((2, 4, 5)).astype(numpy.float32),
            lambda rng: rng.standard_normal((2, 5, 3)).astype(numpy.float32),
            name="batched_1d",
        )

        # Multiple batch dimensions: (B1, B2, M, K) @ (B1, B2, K, N) -> (B1, B2, M, N)
        yield OperationTestConfig(
            jnp.matmul,
            lambda rng: rng.standard_normal((2, 3, 4, 5)).astype(numpy.float32),
            lambda rng: rng.standard_normal((2, 3, 5, 6)).astype(numpy.float32),
            name="batched_2d",
        )

        # Three batch dimensions
        yield OperationTestConfig(
            jnp.matmul,
            lambda rng: rng.standard_normal((2, 2, 2, 3, 4)).astype(numpy.float32),
            lambda rng: rng.standard_normal((2, 2, 2, 4, 5)).astype(numpy.float32),
            name="batched_3d",
        )

        # Test with einsum for more explicit control
        yield OperationTestConfig(
            lambda x, y: jnp.einsum("bij,bjk->bik", x, y),
            lambda rng: rng.standard_normal((3, 4, 5)).astype(numpy.float32),
            lambda rng: rng.standard_normal((3, 5, 6)).astype(numpy.float32),
            name="einsum_batched",
        )

        # Einsum with two batch dimensions
        yield OperationTestConfig(
            lambda x, y: jnp.einsum("abij,abjk->abik", x, y),
            lambda rng: rng.standard_normal((2, 3, 4, 5)).astype(numpy.float32),
            lambda rng: rng.standard_normal((2, 3, 5, 6)).astype(numpy.float32),
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
            lambda rng: rng.standard_normal((3, 4, 5)).astype(numpy.float32),
            lambda rng: rng.standard_normal((3, 5, 6)).astype(numpy.float32),
            name="dot_general_batched",
        )

        # lax.dot_general with two batch dimensions
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
            ),
            lambda rng: rng.standard_normal((2, 3, 4, 5)).astype(numpy.float32),
            lambda rng: rng.standard_normal((2, 3, 5, 6)).astype(numpy.float32),
            name="dot_general_batched_2d",
        )

        # lax.dot_general: contract on first dimension of lhs (transposed lhs)
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
            ),
            lambda rng: rng.standard_normal((3, 5, 4)).astype(
                numpy.float32
            ),  # (B, K, M)
            lambda rng: rng.standard_normal((3, 5, 6)).astype(
                numpy.float32
            ),  # (B, K, N)
            name="dot_general_lhs_transposed",
        )

        # lax.dot_general: contract on last dimension of rhs (transposed rhs)
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            ),
            lambda rng: rng.standard_normal((3, 4, 5)).astype(
                numpy.float32
            ),  # (B, M, K)
            lambda rng: rng.standard_normal((3, 6, 5)).astype(
                numpy.float32
            ),  # (B, N, K)
            name="dot_general_rhs_transposed",
        )

        # lax.dot_general: both transposed
        yield OperationTestConfig(
            lambda x, y: lax.dot_general(
                x,
                y,
                dimension_numbers=(((1,), (2,)), ((0,), (0,))),
            ),
            lambda rng: rng.standard_normal((3, 5, 4)).astype(
                numpy.float32
            ),  # (B, K, M)
            lambda rng: rng.standard_normal((3, 6, 5)).astype(
                numpy.float32
            ),  # (B, N, K)
            name="dot_general_both_transposed",
        )

        # Edge case: zero batch size (empty batch dimension)
        # CPU handles this correctly, returning empty arrays with the right shape.
        # MPS doesn't support zero-sized tensors.
        yield pytest.param(
            OperationTestConfig(
                jnp.matmul,
                numpy.zeros((0, 3, 4), dtype=numpy.float32),
                numpy.zeros((0, 4, 5), dtype=numpy.float32),
                name="batched_zero_batch",
            ),
            marks=[xfail_match("Zero-sized tensors are not supported by MPS")],
        )
