from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig, complex_standard_normal


def make_misc_op_configs():
    with OperationTestConfig.module_name("misc"):
        return [
            # This tests transfer of data with non-contiguous arrays.
            OperationTestConfig(
                lambda x: x,
                lambda key: random.normal(key, (4, 5, 6, 8)).transpose((2, 0, 1, 3)),
            ),
            OperationTestConfig(
                lambda x: jnp.fft.fft(x),
                lambda key: complex_standard_normal(key, (16,), complex=True),
                name="fft-jnp-1d",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.fft(x, axis=0),
                lambda key: complex_standard_normal(key, (8, 4), complex=True),
                name="fft-jnp-axis0",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.ifft(x, axis=1),
                lambda key: complex_standard_normal(key, (4, 8), complex=True),
                name="ifft-jnp-axis1",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.fftn(x, s=(4, 8), axes=(1, 2)),
                lambda key: complex_standard_normal(key, (2, 4, 8), complex=True),
                name="fftn-jnp-axes12",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.ifftn(x, s=(4, 8), axes=(1, 2)),
                lambda key: complex_standard_normal(key, (2, 4, 8), complex=True),
                name="ifftn-jnp-axes12",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.rfftn(x, s=(4, 8), axes=(1, 2)),
                lambda key: random.normal(key, (2, 4, 8)),
                name="rfftn-jnp-axes12",
            ),
            OperationTestConfig(
                lambda x: jnp.fft.irfftn(x, s=(4, 8), axes=(1, 2)),
                lambda key: complex_standard_normal(key, (2, 4, 5), complex=True),
                name="irfftn-jnp-axes12",
            ),
            # FFT variants. Use lax.fft to target stablehlo.fft directly.
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.FFT, (16,)),
                lambda key: complex_standard_normal(key, (3, 16), complex=True),
                name="fft-c2c-1d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.FFT, (4, 8)),
                lambda key: complex_standard_normal(key, (2, 4, 8), complex=True),
                name="fft-c2c-2d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IFFT, (16,)),
                lambda key: complex_standard_normal(key, (3, 16), complex=True),
                name="ifft-c2c-1d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IFFT, (4, 8)),
                lambda key: complex_standard_normal(key, (2, 4, 8), complex=True),
                name="ifft-c2c-2d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.RFFT, (16,)),
                lambda key: random.normal(key, (3, 16)),
                name="rfft-r2c-1d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.RFFT, (4, 8)),
                lambda key: random.normal(key, (2, 4, 8)),
                name="rfft-r2c-2d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IRFFT, (16,)),
                lambda key: complex_standard_normal(key, (3, 9), complex=True),
                name="irfft-c2r-1d",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IRFFT, (4, 8)),
                lambda key: complex_standard_normal(key, (2, 4, 5), complex=True),
                name="irfft-c2r-2d",
            ),
            # Odd-sized FFT tests
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.FFT, (15,)),
                lambda key: complex_standard_normal(key, (3, 15), complex=True),
                name="fft-c2c-1d-odd",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IFFT, (15,)),
                lambda key: complex_standard_normal(key, (3, 15), complex=True),
                name="ifft-c2c-1d-odd",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.RFFT, (15,)),
                lambda key: random.normal(key, (3, 15)),
                name="rfft-r2c-1d-odd",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IRFFT, (15,)),
                lambda key: complex_standard_normal(key, (3, 8), complex=True),
                name="irfft-c2r-1d-odd",
            ),
            OperationTestConfig(
                lambda x: lax.fft(x, lax.FftType.IRFFT, (4, 7)),
                lambda key: complex_standard_normal(key, (2, 4, 4), complex=True),
                name="irfft-c2r-2d-odd",
            ),
        ]
