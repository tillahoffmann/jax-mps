import jax
from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def _all_reduce_pmean(x):
    """Single-device pmean, lowering to stablehlo.all_reduce (issue #201).

    pmap ignores jax.default_device, so pin it to the harness's current default
    device; otherwise the collective always runs on the default (MPS) backend and
    the CPU comparison pass fails its device-placement assertion.
    """
    # getattr: jax.config exposes jax_default_device at runtime but pyright's
    # stubs don't declare it.
    dev = getattr(jax.config, "jax_default_device", None) or jax.devices()[0]
    return jax.pmap(
        lambda y: lax.pmean(y, axis_name="i"), axis_name="i", devices=[dev]
    )(x)


def _partition_id_axis_index(x):
    """Single-device axis_index, lowering to stablehlo.partition_id.

    With one process and one partition, partition_id is 0; therefore the
    pmapped axis index is 0 and this function is an identity.
    """
    dev = getattr(jax.config, "jax_default_device", None) or jax.devices()[0]
    return jax.pmap(lambda y: y + lax.axis_index("i"), axis_name="i", devices=[dev])(x)


def make_collectives_op_configs():
    with OperationTestConfig.module_name("collectives"):
        # stablehlo.all_reduce: single-device identity fallback (issue #201).
        # pmap over one device emits all_reduce with a 1x1 replica_groups; the
        # handler forwards operand -> result. The pmean VJP is itself an
        # all_reduce, so the default grad path exercises the handler on the
        # backward pass too.
        yield OperationTestConfig(
            _all_reduce_pmean,
            lambda key: random.normal(key, (1, 4)),
            name="all_reduce-pmean-identity",
        )
        # stablehlo.partition_id: single-partition zero fallback. The StableHLO
        # op is operand-free by spec; this test covers the normal JAX lowering
        # via pmap axis_index rather than malformed IR validation.
        yield OperationTestConfig(
            _partition_id_axis_index,
            lambda key: jnp.arange(4, dtype=jnp.int32).reshape(1, 4),
            name="partition_id-axis_index-zero",
        )
