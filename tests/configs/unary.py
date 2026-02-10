import numpy
from jax import lax
from jax import numpy as jnp
from jax.scipy import special

from .util import OperationTestConfig, complex_standard_normal


def make_unary_op_configs():
    for complex in [False, True]:
        with OperationTestConfig.module_name(
            "unary-complex" if complex else "unary-real"
        ):
            yield from [
                OperationTestConfig(jnp.abs, complex_standard_normal((17,), complex)),
                OperationTestConfig(jnp.cos, complex_standard_normal((17,), complex)),
                OperationTestConfig(jnp.exp, complex_standard_normal((17,), complex)),
                OperationTestConfig(jnp.flip, complex_standard_normal((17,), complex)),
                # FIXME: Flip ops should probably be in './shape.py'.
                OperationTestConfig(
                    jnp.fliplr, complex_standard_normal((17, 13), complex)
                ),
                OperationTestConfig(
                    jnp.flipud, complex_standard_normal((17, 13), complex)
                ),
                OperationTestConfig(
                    jnp.negative, complex_standard_normal((17,), complex)
                ),
                OperationTestConfig(jnp.sign, complex_standard_normal((17,), complex)),
                OperationTestConfig(jnp.sin, complex_standard_normal((17,), complex)),
                OperationTestConfig(
                    jnp.square, complex_standard_normal((17,), complex)
                ),
                OperationTestConfig(jnp.tan, complex_standard_normal((17,), complex)),
                OperationTestConfig(jnp.tanh, complex_standard_normal((17,), complex)),
                # FIXME: Transpose ops should probably live in './shape.py'.
                OperationTestConfig(
                    jnp.transpose, complex_standard_normal((17, 8, 9), complex)
                ),
                OperationTestConfig(
                    jnp.transpose,
                    complex_standard_normal((17, 8, 9), complex),
                    (1, 0, 2),
                    static_argnums=(1,),
                ),
                OperationTestConfig(jnp.real, complex_standard_normal((17,), complex)),
                OperationTestConfig(jnp.imag, complex_standard_normal((17,), complex)),
                OperationTestConfig(jnp.log, complex_standard_normal((17,), complex)),
                OperationTestConfig(
                    jnp.log1p,
                    0.5 * complex_standard_normal((17,), complex)
                    if complex
                    else numpy.random.gamma(5, size=(17,)) - 1,
                ),
                OperationTestConfig(jnp.sqrt, complex_standard_normal((17,), complex)),
            ]
    yield from [
        OperationTestConfig(jnp.ceil, numpy.random.standard_normal((17,))),
        OperationTestConfig(jnp.floor, numpy.random.standard_normal((17,))),
        OperationTestConfig(jnp.round, numpy.random.standard_normal((17,))),
    ]

    # Ops that don't trivially generalize across real/complex.
    yield from [
        OperationTestConfig(jnp.isfinite, numpy.asarray([0, jnp.nan, jnp.inf])),
        OperationTestConfig(
            jnp.isfinite, numpy.asarray([1, jnp.nan, 1 + 1j * jnp.inf, -jnp.inf + 1j])
        ),
        OperationTestConfig(lax.rsqrt, numpy.random.gamma(5, size=(17,))),
        OperationTestConfig(
            jnp.arcsin,
            numpy.random.uniform(-0.9, 0.9, (17,)).astype(numpy.float32),
        ),
        OperationTestConfig(
            jnp.arccos,
            numpy.random.uniform(-0.9, 0.9, (17,)).astype(numpy.float32),
        ),
        OperationTestConfig(
            jnp.sinh,
            numpy.random.standard_normal((17,)).astype(numpy.float32),
        ),
        OperationTestConfig(
            jnp.cosh,
            numpy.random.standard_normal((17,)).astype(numpy.float32),
        ),
        OperationTestConfig(
            jnp.arcsinh,
            numpy.random.standard_normal((17,)).astype(numpy.float32),
        ),
        OperationTestConfig(
            jnp.arccosh,
            (1 + numpy.random.gamma(5, size=(17,))).astype(numpy.float32),
        ),
        OperationTestConfig(
            jnp.arctanh,
            numpy.random.uniform(-0.9, 0.9, (17,)).astype(numpy.float32),
        ),
        OperationTestConfig(
            jnp.cbrt,
            numpy.random.standard_normal((17,)).astype(numpy.float32),
        ),
        OperationTestConfig(
            special.erfinv,
            numpy.random.uniform(-0.9, 0.9, (17,)).astype(numpy.float32),
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
