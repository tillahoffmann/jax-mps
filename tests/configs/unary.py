import numpy
from jax import lax, random
from jax import numpy as jnp
from jax.scipy import special

from .util import OperationTestConfig, complex_standard_normal


def make_unary_op_configs():
    for complex in [False, True]:
        with OperationTestConfig.module_name(
            "unary-complex" if complex else "unary-real"
        ):
            yield from [
                OperationTestConfig(
                    jnp.abs,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.cos,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.exp,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.negative,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.sign,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.sin,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.square,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.tan,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.tanh,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.real,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.imag,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.log,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.log1p,
                    lambda key, complex=complex: (
                        0.5 * complex_standard_normal(key, (17,), complex)
                        if complex
                        else random.gamma(key, 5.0, (17,)) - 1
                    ),
                ),
                OperationTestConfig(
                    jnp.sqrt,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (17,), complex
                    ),
                ),
            ]
    yield from [
        OperationTestConfig(jnp.ceil, lambda key: random.normal(key, (17,))),
        OperationTestConfig(jnp.floor, lambda key: random.normal(key, (17,))),
        OperationTestConfig(jnp.round, lambda key: random.normal(key, (17,))),
    ]

    # Ops that don't trivially generalize across real/complex.
    yield from [
        OperationTestConfig(jnp.isfinite, numpy.asarray([0, jnp.nan, jnp.inf])),
        OperationTestConfig(
            jnp.isfinite, numpy.asarray([1, jnp.nan, 1 + 1j * jnp.inf, -jnp.inf + 1j])
        ),
        OperationTestConfig(lax.rsqrt, lambda key: random.gamma(key, 5.0, (17,))),
        OperationTestConfig(
            jnp.arcsin,
            lambda key: random.uniform(key, (17,), minval=-0.9, maxval=0.9),
        ),
        OperationTestConfig(
            jnp.arccos,
            lambda key: random.uniform(key, (17,), minval=-0.9, maxval=0.9),
        ),
        OperationTestConfig(
            jnp.sinh,
            lambda key: random.normal(key, (17,)),
        ),
        OperationTestConfig(
            jnp.cosh,
            lambda key: random.normal(key, (17,)),
        ),
        OperationTestConfig(
            jnp.arcsinh,
            lambda key: random.normal(key, (17,)),
        ),
        OperationTestConfig(
            jnp.arccosh,
            lambda key: 1 + random.gamma(key, 5.0, (17,)),
        ),
        OperationTestConfig(
            jnp.arctanh,
            lambda key: random.uniform(key, (17,), minval=-0.9, maxval=0.9),
        ),
        OperationTestConfig(
            jnp.cbrt,
            lambda key: random.normal(key, (17,)),
        ),
        OperationTestConfig(
            special.erfinv,
            lambda key: random.uniform(key, (17,), minval=-0.9, maxval=0.9),
        ),
        OperationTestConfig(
            jnp.logical_not,
            numpy.asarray([True, False, True, False]),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            jnp.invert,
            numpy.array([0, 1, -1, 127, -128], dtype=numpy.int32),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            jnp.invert,
            numpy.array([0, 1, 127, 255, 0x80000000], dtype=numpy.uint32),
            differentiable_argnums=(),
            name="invert-uint32",
        ),
        OperationTestConfig(
            lax.population_count,
            numpy.asarray([0, 1, 3, 7, 15, 127, 128, 255], dtype=numpy.uint8),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lax.population_count,
            numpy.asarray([0, 1, -1, -2, -3, 64, 127, -128], dtype=numpy.int8),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lax.population_count,
            numpy.asarray([0, 1, 0x00FF, 0x0F0F, 0x8000, 0xFFFF], dtype=numpy.uint16),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lax.population_count,
            numpy.asarray([0, 1, -1, -2, -32768, 32767], dtype=numpy.int16),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lax.population_count,
            numpy.asarray(
                [0, 1, 3, 7, 255, 256, 0x80000000, 0xFFFFFFFF], dtype=numpy.uint32
            ),
            differentiable_argnums=(),
        ),
        OperationTestConfig(
            lax.population_count,
            numpy.asarray(
                [0, 1, -1, -2, -2147483648, 2147483647, 0x55555555, 0x33333333],
                dtype=numpy.int32,
            ),
            differentiable_argnums=(),
        ),
    ]
