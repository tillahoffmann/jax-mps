import jax
from jax import numpy as jnp
from jax import random

from jax_plugins.mps.ops import gelu, layer_norm, rms_norm, rope, sdpa

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

        # GELU via jax.nn.gelu (tests plugin monkey-patching)
        yield OperationTestConfig(
            lambda x: jax.nn.gelu(x, approximate=True),
            lambda key: random.normal(key, (4, 8)),
            name="jax_nn_gelu",
        )

        # Layer norm
        yield OperationTestConfig(
            lambda x, w, b: layer_norm(x, w, b, eps=1e-5),
            lambda key: random.normal(key, (2, 4, 8)),
            lambda key: jnp.ones(8),
            lambda key: jnp.zeros(8),
            name="layer_norm",
        )

        # Layer norm (non-trivial eps)
        yield OperationTestConfig(
            lambda x, w, b: layer_norm(x, w, b, eps=1e-12),
            lambda key: random.normal(key, (3, 16)),
            lambda key: random.normal(key, (16,)) * 0.5 + 1.0,
            lambda key: random.normal(key, (16,)) * 0.1,
            name="layer_norm_eps",
        )

        # SDPA with boolean mask (non-trivial: some positions masked)
        yield OperationTestConfig(
            lambda q, k, v: sdpa(
                q,
                k,
                v,
                scale=0.25,
                mask=jnp.array([[[[True, True, False, True]]]]),
            ),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            name="sdpa_masked",
        )

        # RoPE with dynamic offset (passed as JAX value)
        yield OperationTestConfig(
            lambda x, off: rope(x, dims=8, base=10000.0, offset=off),
            lambda key: random.normal(key, (1, 2, 4, 8)),
            lambda key: jnp.int32(3),
            name="rope_dynamic_offset",
            differentiable_argnums=(0,),  # offset is non-differentiable
        )
