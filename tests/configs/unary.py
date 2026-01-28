import numpy
from jax import lax
from jax import numpy as jnp
from jax.scipy import special

from .util import OperationTestConfig


def make_unary_op_configs():
    with OperationTestConfig.module_name("unary"):
        return [
            OperationTestConfig(jnp.abs, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.ceil, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.cos, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.exp, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.flip, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.fliplr, numpy.random.normal(size=(17, 13))),
            OperationTestConfig(jnp.flipud, numpy.random.normal(size=(17, 13))),
            OperationTestConfig(jnp.floor, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.isfinite, numpy.asarray([0, jnp.nan, jnp.inf])),
            OperationTestConfig(jnp.log, numpy.random.gamma(5, size=(17,))),
            OperationTestConfig(jnp.log1p, numpy.random.gamma(5, size=(17,)) - 1),
            OperationTestConfig(jnp.negative, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.sign, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.sin, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.square, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.sqrt, numpy.random.gamma(5, size=(17,))),
            OperationTestConfig(jnp.tan, numpy.random.normal(size=(17,))),
            OperationTestConfig(jnp.tanh, numpy.random.normal(size=(17,))),
            OperationTestConfig(lax.rsqrt, numpy.random.gamma(5, size=(17,))),
            OperationTestConfig(
                special.erfinv,
                numpy.random.uniform(-0.9, 0.9, (17,)).astype(numpy.float32),
            ),
            OperationTestConfig(jnp.transpose, numpy.random.normal(size=(17, 8, 9))),
            OperationTestConfig(
                jnp.transpose,
                numpy.random.normal(size=(17, 8, 9)),
                (1, 0, 2),
                static_argnums=(1,),
            ),
        ]
