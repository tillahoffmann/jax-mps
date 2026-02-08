import numpy
from jax import numpy as jnp

from .util import OperationTestConfig


def make_misc_op_configs():
    with OperationTestConfig.module_name("misc"):
        return [
            # This tests transfer of data with non-contiguous arrays.
            OperationTestConfig(
                lambda x: x,
                numpy.random.standard_normal((4, 5, 6, 8)).transpose((2, 0, 1, 3)),
            ),
            OperationTestConfig(
                lambda x: jnp.fft.fft(x),
                (
                    numpy.random.standard_normal((16,))
                    + 1j * numpy.random.standard_normal((16,))
                ).astype(numpy.complex64),
                differentiable_argnums=(),
            ),
        ]
