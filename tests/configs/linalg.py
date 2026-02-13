import numpy
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

from .util import OperationTestConfig


def _random_posdef(
    rng: numpy.random.Generator, n: int, batch_shape: tuple[int, ...] = ()
) -> numpy.ndarray:
    """Generate random positive-definite matrices of shape (*batch_shape, n, n)."""
    shape = (*batch_shape, n, n)
    A = rng.standard_normal(shape).astype(numpy.float32)
    # A @ A.T for batched inputs: (..., n, n) @ (..., n, n) -> (..., n, n)
    result = numpy.einsum("...ij,...kj->...ik", A, A) + n * numpy.eye(
        n, dtype=numpy.float32
    )
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
    rng: numpy.random.Generator,
    n: int,
    lower: bool = True,
    batch_shape: tuple[int, ...] = (),
) -> numpy.ndarray:
    """Generate random well-conditioned triangular matrices of shape (*batch_shape, n, n)."""
    shape = (*batch_shape, n, n)
    M = rng.standard_normal(shape).astype(numpy.float32)
    L = numpy.tril(M) if lower else numpy.triu(M)
    # Fix diagonal to ensure well-conditioned: |diag| + 1
    diag_idx = numpy.arange(n)
    L[..., diag_idx, diag_idx] = numpy.abs(L[..., diag_idx, diag_idx]) + 1
    return L


def _random_triangular_unit_diag(rng: numpy.random.Generator, n: int) -> numpy.ndarray:
    M = rng.standard_normal((n, n)).astype(numpy.float32)
    L = numpy.tril(M)
    numpy.fill_diagonal(L, 1.0)
    return L


def make_linalg_op_configs():
    with OperationTestConfig.module_name("linalg"):
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                lambda rng, n=n: _random_posdef(rng, n),
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
                lambda rng, n=n: _random_triangular(rng, n, lower=True),
                lambda rng, n=n: rng.standard_normal((n, 1)).astype(numpy.float32),
                name=f"triangular_solve_lower_{n}x{n}",
            )
            # Upper triangular, single RHS column.
            yield OperationTestConfig(
                _solve_triangular_upper,
                lambda rng, n=n: _random_triangular(rng, n, lower=False),
                lambda rng, n=n: rng.standard_normal((n, 1)).astype(numpy.float32),
                name=f"triangular_solve_upper_{n}x{n}",
            )
            # Lower triangular, multiple RHS columns.
            yield OperationTestConfig(
                _solve_triangular_lower,
                lambda rng, n=n: _random_triangular(rng, n, lower=True),
                lambda rng, n=n: rng.standard_normal((n, 3)).astype(numpy.float32),
                name=f"triangular_solve_lower_{n}x{n}_multi_rhs",
            )
            # Upper triangular, multiple RHS columns.
            yield OperationTestConfig(
                _solve_triangular_upper,
                lambda rng, n=n: _random_triangular(rng, n, lower=False),
                lambda rng, n=n: rng.standard_normal((n, 3)).astype(numpy.float32),
                name=f"triangular_solve_upper_{n}x{n}_multi_rhs",
            )

        # Transpose: solve L^T x = b and U^T x = b
        yield OperationTestConfig(
            _solve_triangular_lower_trans,
            lambda rng: _random_triangular(rng, 3, lower=True),
            lambda rng: rng.standard_normal((3, 1)).astype(numpy.float32),
            name="triangular_solve_lower_trans",
        )
        yield OperationTestConfig(
            _solve_triangular_upper_trans,
            lambda rng: _random_triangular(rng, 3, lower=False),
            lambda rng: rng.standard_normal((3, 1)).astype(numpy.float32),
            name="triangular_solve_upper_trans",
        )

        # Unit diagonal: assume diagonal elements are 1
        yield OperationTestConfig(
            _solve_triangular_unit_diag,
            lambda rng: _random_triangular_unit_diag(rng, 3),
            lambda rng: rng.standard_normal((3, 1)).astype(numpy.float32),
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

        # Batched inputs - gradient testing disabled because dot_general
        # doesn't fully support batch dimensions yet.
        for batch_shape in [(2,), (2, 3)]:
            batch_str = "x".join(map(str, batch_shape))
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                lambda rng, bs=batch_shape: _random_posdef(rng, 3, batch_shape=bs),
                differentiable_argnums=(),
                name=f"cholesky_batched_{batch_str}",
            )
            yield OperationTestConfig(
                _solve_triangular_lower,
                lambda rng, bs=batch_shape: _random_triangular(
                    rng, 3, lower=True, batch_shape=bs
                ),
                lambda rng, bs=batch_shape: rng.standard_normal((*bs, 3, 1)).astype(
                    numpy.float32
                ),
                differentiable_argnums=(),
                name=f"triangular_solve_batched_{batch_str}",
            )
