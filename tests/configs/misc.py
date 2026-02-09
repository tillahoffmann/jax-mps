import numpy
from jax import lax
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
                name="fft-jnp-1d",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.fft(x, axis=0),
                (
                    numpy.random.standard_normal((8, 4))
                    + 1j * numpy.random.standard_normal((8, 4))
                ).astype(numpy.complex64),
                name="fft-jnp-axis0",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.ifft(x, axis=1),
                (
                    numpy.random.standard_normal((4, 8))
                    + 1j * numpy.random.standard_normal((4, 8))
                ).astype(numpy.complex64),
                name="ifft-jnp-axis1",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.fftn(x, s=(4, 8), axes=(1, 2)),
                (
                    numpy.random.standard_normal((2, 4, 8))
                    + 1j * numpy.random.standard_normal((2, 4, 8))
                ).astype(numpy.complex64),
                name="fftn-jnp-axes12",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.ifftn(x, s=(4, 8), axes=(1, 2)),
                (
                    numpy.random.standard_normal((2, 4, 8))
                    + 1j * numpy.random.standard_normal((2, 4, 8))
                ).astype(numpy.complex64),
                name="ifftn-jnp-axes12",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.rfftn(x, s=(4, 8), axes=(1, 2)),
                numpy.random.standard_normal((2, 4, 8)).astype(numpy.float32),
                name="rfftn-jnp-axes12",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.irfftn(x, s=(4, 8), axes=(1, 2)),
                (
                    numpy.random.standard_normal((2, 4, 5))
                    + 1j * numpy.random.standard_normal((2, 4, 5))
                ).astype(numpy.complex64),
                name="irfftn-jnp-axes12",
            ),
            # FFT variants. Use lax.fft to target stablehlo.fft directly.
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.FFT, (16,)),
                (
                    numpy.random.standard_normal((3, 16))
                    + 1j * numpy.random.standard_normal((3, 16))
                ).astype(numpy.complex64),
                name="fft-c2c-1d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.FFT, (4, 8)),
                (
                    numpy.random.standard_normal((2, 4, 8))
                    + 1j * numpy.random.standard_normal((2, 4, 8))
                ).astype(numpy.complex64),
                name="fft-c2c-2d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IFFT, (16,)),
                (
                    numpy.random.standard_normal((3, 16))
                    + 1j * numpy.random.standard_normal((3, 16))
                ).astype(numpy.complex64),
                name="ifft-c2c-1d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IFFT, (4, 8)),
                (
                    numpy.random.standard_normal((2, 4, 8))
                    + 1j * numpy.random.standard_normal((2, 4, 8))
                ).astype(numpy.complex64),
                name="ifft-c2c-2d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.RFFT, (16,)),
                numpy.random.standard_normal((3, 16)).astype(numpy.float32),
                name="rfft-r2c-1d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.RFFT, (4, 8)),
                numpy.random.standard_normal((2, 4, 8)).astype(numpy.float32),
                name="rfft-r2c-2d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IRFFT, (16,)),
                (
                    numpy.random.standard_normal((3, 9))
                    + 1j * numpy.random.standard_normal((3, 9))
                ).astype(numpy.complex64),
                name="irfft-c2r-1d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IRFFT, (4, 8)),
                (
                    numpy.random.standard_normal((2, 4, 5))
                    + 1j * numpy.random.standard_normal((2, 4, 5))
                ).astype(numpy.complex64),
                name="irfft-c2r-2d",
            ),
            # Odd-sized FFT tests (for roundToOddHermitean handling)
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.RFFT, (15,)),
                numpy.random.standard_normal((3, 15)).astype(numpy.float32),
                name="rfft-r2c-1d-odd",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IRFFT, (15,)),
                (
                    numpy.random.standard_normal((3, 8))
                    + 1j * numpy.random.standard_normal((3, 8))
                ).astype(numpy.complex64),
                name="irfft-c2r-1d-odd",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IRFFT, (4, 7)),
                (
                    numpy.random.standard_normal((2, 4, 4))
                    + 1j * numpy.random.standard_normal((2, 4, 4))
                ).astype(numpy.complex64),
                name="irfft-c2r-2d-odd",
            ),
        ]
