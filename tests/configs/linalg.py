import numpy
import pytest
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

from .util import OperationTestConfig


def _random_posdef(n):
    """Generate a random positive-definite matrix of size n x n."""
    A = numpy.random.standard_normal((n, n)).astype(numpy.float32)
    return A @ A.T + n * numpy.eye(n, dtype=numpy.float32)


def _solve_triangular_lower(L, B):
    return solve_triangular(L, B, lower=True)


def _solve_triangular_upper(U, B):
    return solve_triangular(U, B, lower=False)


def _random_triangular(n, lower=True):
    """Generate a random well-conditioned triangular matrix."""
    M = numpy.random.standard_normal((n, n)).astype(numpy.float32)
    L = numpy.tril(M) if lower else numpy.triu(M)
    numpy.fill_diagonal(L, numpy.abs(numpy.diag(L)) + 1)
    return L


def make_linalg_op_configs():
    with OperationTestConfig.module_name("linalg"):
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                _random_posdef(n),
                name=f"cholesky_{n}x{n}",
            )

        # Cholesky on a non-positive-definite matrix (should match CPU NaN behavior).
        yield OperationTestConfig(
            jnp.linalg.cholesky,
            numpy.array([[-1, 0], [0, 1]], dtype=numpy.float32),
            name="cholesky_non_posdef",
        )

        for n in [2, 3, 4]:
            # Lower triangular, single RHS column.
            yield OperationTestConfig(
                _solve_triangular_lower,
                _random_triangular(n, lower=True),
                numpy.random.standard_normal((n, 1)).astype(numpy.float32),
                name=f"triangular_solve_lower_{n}x{n}",
            )
            # Upper triangular, single RHS column.
            yield OperationTestConfig(
                _solve_triangular_upper,
                _random_triangular(n, lower=False),
                numpy.random.standard_normal((n, 1)).astype(numpy.float32),
                name=f"triangular_solve_upper_{n}x{n}",
            )
            # Lower triangular, multiple RHS columns.
            yield OperationTestConfig(
                _solve_triangular_lower,
                _random_triangular(n, lower=True),
                numpy.random.standard_normal((n, 3)).astype(numpy.float32),
                name=f"triangular_solve_lower_{n}x{n}_multi_rhs",
            )

        # Batched inputs: not yet supported by native MPS kernels.
        yield pytest.param(
            OperationTestConfig(
                jnp.linalg.cholesky,
                _random_posdef(3)[None, :, :].repeat(2, axis=0),
                name="cholesky_batched",
            ),
            marks=pytest.mark.xfail(reason="Batched linalg not yet supported"),
        )
