import numpy
from jax import numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular

from .util import OperationTestConfig


def _random_posdef(key, n: int, batch_shape: tuple[int, ...] = ()):
    """Generate random positive-definite matrices of shape (*batch_shape, n, n)."""
    shape = (*batch_shape, n, n)
    A = random.normal(key, shape)
    # A @ A.T for batched inputs: (..., n, n) @ (..., n, n) -> (..., n, n)
    result = jnp.einsum("...ij,...kj->...ik", A, A) + n * jnp.eye(n, dtype=jnp.float32)
    return result


def _solve_triangular_lower(L, B):
    return solve_triangular(L, B, lower=True)


def _solve_triangular_upper(U, B):
    return solve_triangular(U, B, lower=False)


def _solve_triangular_lower_trans(L, B):
    return solve_triangular(L, B, lower=True, trans=1)


def _solve_triangular_upper_trans(U, B):
    return solve_triangular(U, B, lower=False, trans=1)


def _solve_triangular_unit_diag(L, B):
    return solve_triangular(L, B, lower=True, unit_diagonal=True)


def _random_triangular(
    key,
    n: int,
    lower: bool = True,
    batch_shape: tuple[int, ...] = (),
):
    """Generate random well-conditioned triangular matrices of shape (*batch_shape, n, n)."""
    shape = (*batch_shape, n, n)
    M = random.normal(key, shape)
    L = jnp.tril(M) if lower else jnp.triu(M)
    # Fix diagonal to ensure well-conditioned: |diag| + 1
    # Use eye mask to modify diagonal without advanced indexing
    eye = jnp.eye(n, dtype=jnp.float32)
    # Get off-diagonal elements by masking out diagonal
    off_diag = L * (1 - eye)
    # For diagonal, take abs and add 1 (using the diagonal part of L)
    diag_values = jnp.abs(L * eye) + eye
    return off_diag + diag_values


def _random_triangular_unit_diag(key, n: int):
    """Generate random unit-diagonal triangular matrix."""
    M = random.normal(key, (n, n))
    L = jnp.tril(M)
    # Use eye mask to set diagonal to 1 without advanced indexing
    eye = jnp.eye(n, dtype=jnp.float32)
    return L * (1 - eye) + eye


def make_linalg_op_configs():
    with OperationTestConfig.module_name("linalg"):
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                lambda key, n=n: _random_posdef(key, n),
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
                lambda key, n=n: _random_triangular(key, n, lower=True),
                lambda key, n=n: random.normal(key, (n, 1)),
                name=f"triangular_solve_lower_{n}x{n}",
            )
            # Upper triangular, single RHS column.
            yield OperationTestConfig(
                _solve_triangular_upper,
                lambda key, n=n: _random_triangular(key, n, lower=False),
                lambda key, n=n: random.normal(key, (n, 1)),
                name=f"triangular_solve_upper_{n}x{n}",
            )
            # Lower triangular, multiple RHS columns.
            yield OperationTestConfig(
                _solve_triangular_lower,
                lambda key, n=n: _random_triangular(key, n, lower=True),
                lambda key, n=n: random.normal(key, (n, 3)),
                name=f"triangular_solve_lower_{n}x{n}_multi_rhs",
            )
            # Upper triangular, multiple RHS columns.
            yield OperationTestConfig(
                _solve_triangular_upper,
                lambda key, n=n: _random_triangular(key, n, lower=False),
                lambda key, n=n: random.normal(key, (n, 3)),
                name=f"triangular_solve_upper_{n}x{n}_multi_rhs",
            )

        # Transpose: solve L^T x = b and U^T x = b
        yield OperationTestConfig(
            _solve_triangular_lower_trans,
            lambda key: _random_triangular(key, 3, lower=True),
            lambda key: random.normal(key, (3, 1)),
            name="triangular_solve_lower_trans",
        )
        yield OperationTestConfig(
            _solve_triangular_upper_trans,
            lambda key: _random_triangular(key, 3, lower=False),
            lambda key: random.normal(key, (3, 1)),
            name="triangular_solve_upper_trans",
        )

        # Unit diagonal: assume diagonal elements are 1
        yield OperationTestConfig(
            _solve_triangular_unit_diag,
            lambda key: _random_triangular_unit_diag(key, 3),
            lambda key: random.normal(key, (3, 1)),
            name="triangular_solve_unit_diagonal",
        )

        # 1x1 matrices (trivial edge case)
        yield OperationTestConfig(
            jnp.linalg.cholesky,
            numpy.array([[4.0]], dtype=numpy.float32),
            name="cholesky_1x1",
        )
        yield OperationTestConfig(
            _solve_triangular_lower,
            numpy.array([[2.0]], dtype=numpy.float32),
            numpy.array([[6.0]], dtype=numpy.float32),
            name="triangular_solve_1x1",
        )

        # Batched inputs
        for batch_shape in [(2,), (2, 3)]:
            batch_str = "x".join(map(str, batch_shape))
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                lambda key, bs=batch_shape: _random_posdef(key, 3, batch_shape=bs),
                name=f"cholesky_batched_{batch_str}",
            )
            yield OperationTestConfig(
                _solve_triangular_lower,
                lambda key, bs=batch_shape: _random_triangular(
                    key, 3, lower=True, batch_shape=bs
                ),
                lambda key, bs=batch_shape: random.normal(key, (*bs, 3, 1)),
                name=f"triangular_solve_batched_{batch_str}",
            )

        # Edge case: zero batch size (empty batch dimension)
        # CPU handles this correctly, returning empty arrays with the right shape.
        yield OperationTestConfig(
            jnp.linalg.cholesky,
            numpy.zeros((0, 3, 3), dtype=numpy.float32),
            name="cholesky_zero_batch",
        )
        yield OperationTestConfig(
            _solve_triangular_lower,
            numpy.zeros((0, 3, 3), dtype=numpy.float32),
            numpy.zeros((0, 3, 1), dtype=numpy.float32),
            name="triangular_solve_zero_batch",
        )
