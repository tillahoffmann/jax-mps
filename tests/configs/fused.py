import jax
from jax import numpy as jnp
from jax import random

from jax_plugins.mps.ops import gelu, rms_norm, rope, sdpa

from .util import OperationTestConfig


def make_fused_op_configs():
    with OperationTestConfig.module_name("fused"):
        # GELU approximate
        yield OperationTestConfig(
            gelu,
            lambda key: random.normal(key, (4, 8)),
            name="gelu",
        )

        # RMS norm
        yield OperationTestConfig(
            lambda x, w: rms_norm(x, w, eps=1e-6),
            lambda key: random.normal(key, (2, 4, 8)),
            lambda key: jnp.ones(8),
            name="rms_norm",
        )

        # RoPE
        yield OperationTestConfig(
            lambda x: rope(x, dims=8, base=10000.0, offset=0),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            name="rope",
        )

        # RoPE with offset
        yield OperationTestConfig(
            lambda x: rope(x, dims=8, base=10000.0, offset=5),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            name="rope_offset",
        )

        # SDPA (non-causal)
        yield OperationTestConfig(
            lambda q, k, v: sdpa(q, k, v, scale=0.25),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            name="sdpa",
        )

        # SDPA (causal)
        yield OperationTestConfig(
            lambda q, k, v: sdpa(q, k, v, scale=0.25, is_causal=True),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            name="sdpa_causal",
        )

        # GELU via jax.nn.gelu (tests monkey-patching when JAX_MPS_FUSED_OPS=1)
        yield OperationTestConfig(
            lambda x: jax.nn.gelu(x, approximate=True),
            lambda key: random.normal(key, (4, 8)),
            name="jax_nn_gelu",
        )
