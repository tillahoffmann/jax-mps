import numpy
import pytest
from jax import numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular

from .util import OperationTestConfig, xfail_match


def _random_invertible(key, n: int, batch_shape: tuple[int, ...] = ()):
    """Generate random well-conditioned invertible matrices."""
    shape = (*batch_shape, n, n)
    A = random.normal(key, shape)
    # Add n * I to ensure well-conditioned (diagonally dominant)
    return A + n * jnp.eye(n, dtype=jnp.float32)


def _random_symmetric(key, n: int, batch_shape: tuple[int, ...] = ()):
    """Generate random symmetric matrices."""
    shape = (*batch_shape, n, n)
    A = random.normal(key, shape)
    return (A + jnp.swapaxes(A, -2, -1)) / 2


def _svd_values(A):
    """Return singular values only (avoids sign ambiguity in U/Vh)."""
    return jnp.linalg.svd(A, compute_uv=False)


def _svd_reconstruct(A):
    """Reconstruct A from its SVD to verify correctness without sign issues."""
    U, S, Vh = jnp.linalg.svd(A, full_matrices=False)
    return U @ jnp.diag(S) @ Vh


def _qr_r(A):
    """Return R factor only (avoids sign ambiguity in Q)."""
    Q, R = jnp.linalg.qr(A)
    # Normalize sign: make R diagonal positive
    signs = jnp.sign(jnp.diag(R))
    return R * signs[:, None]


def _qr_reconstruct(A):
    """Reconstruct A from its QR to verify correctness without sign issues."""
    Q, R = jnp.linalg.qr(A)
    return Q @ R


def _eigh_values(A):
    """Return eigenvalues only (avoids sign ambiguity in eigenvectors)."""
    return jnp.linalg.eigh(A)[0]


def _eigh_reconstruct(A):
    """Reconstruct A from eigh to verify correctness: A = V @ diag(w) @ V.T."""
    w, V = jnp.linalg.eigh(A)
    return V @ jnp.diag(w) @ V.T


def _lu_solve(A, b):
    """Solve Ax = b using LU factorization."""
    from jax.scipy.linalg import lu_factor, lu_solve

    lu, piv = lu_factor(A)
    return lu_solve((lu, piv), b)


def _slogdet_logabsdet(A):
    """Return log absolute determinant (sign can differ for numerical reasons)."""
    return jnp.linalg.slogdet(A)[1]


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

        # Cholesky on a non-positive-definite matrix.
        # MPS Cholesky returns input unchanged (no error), while CPU returns NaN.
        yield pytest.param(
            OperationTestConfig(
                jnp.linalg.cholesky,
                numpy.array([[-1, 0], [0, 1]], dtype=numpy.float32),
                name="cholesky_non_posdef",
            ),
            marks=[xfail_match("Values are not close")],
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

        # Singular triangular matrix (zero on diagonal): MPS returns NaN (to avoid
        # LAPACK abort), while CPU returns inf. Values differ but no crash.
        yield pytest.param(
            OperationTestConfig(
                _solve_triangular_lower,
                numpy.array([[1.0, 0.0], [1.0, 0.0]], dtype=numpy.float32),
                numpy.array([[1.0], [2.0]], dtype=numpy.float32),
                name="triangular_solve_singular",
            ),
            marks=[xfail_match("Values are not close")],
        )

        # Unit diagonal with zeros on actual diagonal: should NOT be treated as
        # singular because unit_diagonal=True ignores the provided diagonal.
        yield OperationTestConfig(
            _solve_triangular_unit_diag,
            numpy.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 3.0, 0.0]], dtype=numpy.float32
            ),
            numpy.array([[1.0], [2.0], [3.0]], dtype=numpy.float32),
            name="triangular_solve_unit_diag_zero_diagonal",
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

        # --- SVD (needs 'svd'/'eigh' primitives not available on MPS) ---

        _xfail_no_prim = xfail_match(
            "not found for platform mps|custom call target|Output count mismatch"
        )

        # SVD singular values (no sign ambiguity)
        for n in [2, 3, 4]:
            yield pytest.param(
                OperationTestConfig(
                    _svd_values,
                    lambda key, n=n: random.normal(key, (n, n)),
                    name=f"svd_values_{n}x{n}",
                ),
                marks=[_xfail_no_prim],
            )

        # SVD reconstruction (avoids sign ambiguity in U/Vh)
        for n in [2, 3, 4]:
            yield pytest.param(
                OperationTestConfig(
                    _svd_reconstruct,
                    lambda key, n=n: random.normal(key, (n, n)),
                    name=f"svd_reconstruct_{n}x{n}",
                ),
                marks=[_xfail_no_prim],
            )

        # Rectangular SVD
        yield pytest.param(
            OperationTestConfig(
                _svd_values,
                lambda key: random.normal(key, (4, 2)),
                name="svd_values_4x2",
            ),
            marks=[_xfail_no_prim],
        )
        yield pytest.param(
            OperationTestConfig(
                _svd_values,
                lambda key: random.normal(key, (2, 4)),
                name="svd_values_2x4",
            ),
            marks=[_xfail_no_prim],
        )

        # Batched SVD
        yield pytest.param(
            OperationTestConfig(
                _svd_values,
                lambda key: random.normal(key, (2, 3, 3)),
                name="svd_values_batched_2",
            ),
            marks=[_xfail_no_prim],
        )

        # --- QR (needs QR custom_call handler not available on MPS) ---

        # QR R-factor (sign-normalized to avoid ambiguity)
        for n in [2, 3, 4]:
            yield pytest.param(
                OperationTestConfig(
                    _qr_r,
                    lambda key, n=n: random.normal(key, (n, n)),
                    name=f"qr_r_{n}x{n}",
                ),
                marks=[_xfail_no_prim],
            )

        # QR reconstruction
        for n in [2, 3, 4]:
            yield pytest.param(
                OperationTestConfig(
                    _qr_reconstruct,
                    lambda key, n=n: random.normal(key, (n, n)),
                    name=f"qr_reconstruct_{n}x{n}",
                ),
                marks=[_xfail_no_prim],
            )

        # Rectangular QR (tall)
        yield pytest.param(
            OperationTestConfig(
                _qr_reconstruct,
                lambda key: random.normal(key, (4, 2)),
                name="qr_reconstruct_4x2",
            ),
            marks=[_xfail_no_prim],
        )

        # Batched QR
        yield pytest.param(
            OperationTestConfig(
                _qr_reconstruct,
                lambda key: random.normal(key, (2, 3, 3)),
                name="qr_reconstruct_batched_2",
            ),
            marks=[_xfail_no_prim],
        )

        # --- Eigenvalue Decomposition (needs 'eigh'/'eig' primitives) ---

        # Eigenvalues only (no sign ambiguity)
        for n in [2, 3, 4]:
            yield pytest.param(
                OperationTestConfig(
                    _eigh_values,
                    lambda key, n=n: _random_symmetric(key, n),
                    name=f"eigh_values_{n}x{n}",
                ),
                marks=[_xfail_no_prim],
            )

        # Eigh reconstruction (A = V @ diag(w) @ V.T)
        for n in [2, 3, 4]:
            yield pytest.param(
                OperationTestConfig(
                    _eigh_reconstruct,
                    lambda key, n=n: _random_symmetric(key, n),
                    name=f"eigh_reconstruct_{n}x{n}",
                ),
                marks=[_xfail_no_prim],
            )

        # Eigh on positive definite matrix
        yield pytest.param(
            OperationTestConfig(
                _eigh_values,
                lambda key: _random_posdef(key, 3),
                name="eigh_values_posdef",
            ),
            marks=[_xfail_no_prim],
        )

        # Batched eigh
        yield pytest.param(
            OperationTestConfig(
                _eigh_values,
                lambda key: _random_symmetric(key, 3, batch_shape=(2,)),
                name="eigh_values_batched_2",
            ),
            marks=[_xfail_no_prim],
        )

        # General (non-symmetric) eigenvalues
        yield pytest.param(
            OperationTestConfig(
                jnp.linalg.eigvals,
                lambda key: random.normal(key, (3, 3)),
                name="eigvals_3x3",
            ),
            marks=[_xfail_no_prim],
        )

        # --- LU Decomposition ---

        # LU via solve (most practical test: A x = b)
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                _lu_solve,
                lambda key, n=n: _random_invertible(key, n),
                lambda key, n=n: random.normal(key, (n,)),
                name=f"lu_solve_{n}x{n}",
            )

        # LU solve with multiple RHS
        yield OperationTestConfig(
            _lu_solve,
            lambda key: _random_invertible(key, 3),
            lambda key: random.normal(key, (3, 2)),
            name="lu_solve_3x3_multi_rhs",
        )

        # Batched LU solve
        yield OperationTestConfig(
            _lu_solve,
            lambda key: _random_invertible(key, 3, batch_shape=(2,)),
            lambda key: random.normal(key, (2, 3)),
            name="lu_solve_batched_2",
        )

        # --- Linear Solve ---

        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.solve,
                lambda key, n=n: _random_invertible(key, n),
                lambda key, n=n: random.normal(key, (n,)),
                name=f"solve_{n}x{n}",
            )

        # Solve with matrix RHS
        yield OperationTestConfig(
            jnp.linalg.solve,
            lambda key: _random_invertible(key, 3),
            lambda key: random.normal(key, (3, 2)),
            name="solve_3x3_matrix_rhs",
        )

        # Batched solve
        yield OperationTestConfig(
            jnp.linalg.solve,
            lambda key: _random_invertible(key, 3, batch_shape=(2,)),
            lambda key: random.normal(key, (2, 3, 1)),
            name="solve_batched_2",
        )

        # --- Matrix Inverse ---

        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.inv,
                lambda key, n=n: _random_invertible(key, n),
                name=f"inv_{n}x{n}",
            )

        # Batched inverse
        yield OperationTestConfig(
            jnp.linalg.inv,
            lambda key: _random_invertible(key, 3, batch_shape=(2,)),
            name="inv_batched_2",
        )

        # --- Determinant ---

        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.det,
                lambda key, n=n: _random_invertible(key, n),
                name=f"det_{n}x{n}",
            )

        # slogdet (log absolute determinant, avoids sign issues)
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                _slogdet_logabsdet,
                lambda key, n=n: _random_invertible(key, n),
                name=f"slogdet_{n}x{n}",
            )

        # Batched det
        yield OperationTestConfig(
            jnp.linalg.det,
            lambda key: _random_invertible(key, 3, batch_shape=(2,)),
            name="det_batched_2",
        )

        # --- Norms ---

        # Vector norms
        yield OperationTestConfig(
            jnp.linalg.norm,
            lambda key: random.normal(key, (5,)),
            name="norm_vector",
        )

        # Matrix Frobenius norm
        yield OperationTestConfig(
            jnp.linalg.norm,
            lambda key: random.normal(key, (3, 4)),
            name="norm_matrix_fro",
        )

        # --- Matrix Power ---

        yield OperationTestConfig(
            lambda A: jnp.linalg.matrix_power(A, 2),
            lambda key: _random_invertible(key, 3),
            name="matrix_power_2",
        )
        yield OperationTestConfig(
            lambda A: jnp.linalg.matrix_power(A, 3),
            lambda key: _random_invertible(key, 3),
            name="matrix_power_3",
        )
