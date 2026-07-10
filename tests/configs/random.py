from jax import numpy as jnp
from jax import random

from .util import MPS_DEVICE, OperationTestConfig


def make_random_op_configs(partitionable=None):
    """Random-op configs, optionally pinned to a jax_threefry_partitionable setting.

    Pass ``partitionable=True``/``False`` to run every config under that flag
    (applied at evaluation time via ``config_overrides``, since the flag is read
    at lowering time). The two settings select different count-packing lowerings
    around the same threefry2x32 primitive (issue #196), so calling this twice
    checks MPS-vs-CPU equivalence under both. ``None`` keeps the ambient default.
    """
    overrides = (
        None if partitionable is None else {"jax_threefry_partitionable": partitionable}
    )
    suffix = (
        ""
        if partitionable is None
        else ("_partitionable" if partitionable else "_original")
    )

    def _build():
        with OperationTestConfig.module_name("random"):
            # scalar / small-odd / even-2d / large-odd — the last exercises the
            # odd/even split boundary in the count-packing layout (issue #196).
            for shape in [(), (3,), (4, 8), (257,)]:
                yield from [
                    OperationTestConfig(
                        random.normal, random.key(17), shape, static_argnums=(1,)
                    ),
                    OperationTestConfig(
                        random.truncated_normal,
                        random.key(17),
                        -0.1,
                        0.2,
                        shape,
                        static_argnums=(3,),
                    ),
                    OperationTestConfig(
                        random.uniform,
                        random.key(17),
                        (3,),
                        None,
                        0.3,
                        0.5,
                        static_argnums=(1,),
                    ),
                    OperationTestConfig(random.split, random.key(18)),
                    OperationTestConfig(
                        random.split, random.key(18), 5, static_argnums=(1,)
                    ),
                    # Integer outputs are compared MPS-vs-CPU exactly, so these
                    # pin bit-exactness of the fused threefry Metal kernel to
                    # JAX's CPU backend (issue #196). uint8/uint16 exercise the
                    # sub-word packing paths; uint32 the full-word path.
                    OperationTestConfig(
                        lambda key, shape: random.bits(key, shape, dtype=jnp.uint8),
                        random.key(19),
                        shape,
                        static_argnums=(1,),
                        name="bits_uint8",
                    ),
                    OperationTestConfig(
                        lambda key, shape: random.bits(key, shape, dtype=jnp.uint16),
                        random.key(19),
                        shape,
                        static_argnums=(1,),
                        name="bits_uint16",
                    ),
                    OperationTestConfig(
                        random.bits, random.key(19), shape, static_argnums=(1,)
                    ),
                    OperationTestConfig(
                        lambda key, shape: random.randint(key, shape, 0, 1_000_000),
                        random.key(20),
                        shape,
                        static_argnums=(1,),
                        name="randint",
                    ),
                ]
            # Bug: indexing into split result differs on MPS vs CPU in eager mode.
            # JIT compiles around the bug; only eager exposes it.
            if MPS_DEVICE is not None:
                yield OperationTestConfig(
                    lambda key: random.split(key)[0],
                    random.key(42),
                    name="split_index",
                )

    for config in _build():
        if overrides is not None:
            config.config_overrides = overrides
            config.name = f"{config.name}{suffix}"
        yield config
