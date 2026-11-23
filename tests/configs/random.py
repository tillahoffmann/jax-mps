import jax
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
            # (5,) packs 8-bit draws into an even word count; (257,) an odd one.
            for shape in [(), (3,), (5,), (4, 8), (257,)]:
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
                        shape,
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
            # Multi-dim split shape and a large split (counter layout across
            # threadgroups), fused into @mps.threefry_split (issue #210).
            yield OperationTestConfig(
                lambda key: random.split(key, (2, 3)), random.key(21), name="split_2d"
            )
            yield OperationTestConfig(
                lambda key: random.split(key, 1000), random.key(21), name="split_1000"
            )
            # Large draws span many threadgroups of the fused bits kernel.
            for dtype in (jnp.uint8, jnp.uint32):
                yield OperationTestConfig(
                    lambda key, dtype=dtype: random.bits(key, (513, 7), dtype=dtype),
                    random.key(22),
                    name=f"bits_large_{jnp.dtype(dtype).name}",
                )
            # Batched keys keep JAX's default lowering around threefry2x32.
            yield OperationTestConfig(
                lambda keys: jax.vmap(lambda k: random.uniform(k, (3,)))(keys),
                random.split(random.key(23), 4),
                name="uniform_batched_keys",
            )
            yield OperationTestConfig(
                lambda keys: jax.vmap(random.split)(keys),
                random.split(random.key(23), 4),
                name="split_batched_keys",
            )
            yield OperationTestConfig(
                lambda key: random.fold_in(key, 7), random.key(24), name="fold_in"
            )
            # 64-bit draws pair two hashed words per element (needs x64).
            for shape in [(), (257,)]:
                yield OperationTestConfig(
                    lambda key, shape=shape: random.bits(key, shape, dtype=jnp.uint64),
                    random.key(25),
                    name=f"bits_uint64_{len(shape)}d",
                    config_overrides={"jax_enable_x64": True},
                )
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
            config.config_overrides = {**config.config_overrides, **overrides}
            config.name = f"{config.name}{suffix}"
        yield config
