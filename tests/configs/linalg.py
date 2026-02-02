import numpy
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

from .util import OperationTestConfig


def _random_posdef(n):
    """Generate a random positive-definite matrix of size n x n."""
    A = numpy.random.standard_normal((n, n)).astype(numpy.float32)
    return A @ A.T + n * numpy.eye(n, dtype=numpy.float32)


def _solve_triangular(A, B):
    """Wrapper exercising stablehlo.triangular_solve via jax.scipy."""
    L = jnp.linalg.cholesky(A)
    return solve_triangular(L, B, lower=True)


def make_linalg_op_configs():
    with OperationTestConfig.module_name("linalg"):
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                _random_posdef(n),
                differentiable_argnums=(),
                name=f"cholesky_{n}x{n}",
            )

        for n in [2, 3, 4]:
            A = _random_posdef(n)
            B = numpy.random.standard_normal((n, 1)).astype(numpy.float32)
            yield OperationTestConfig(
                _solve_triangular,
                A,
                B,
                differentiable_argnums=(),
                name=f"triangular_solve_{n}x{n}",
            )
