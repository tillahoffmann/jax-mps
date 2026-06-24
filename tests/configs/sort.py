from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def make_sort_op_configs():
    """Test configs for sort, topk, argmax/argmin operations."""
    with OperationTestConfig.module_name("sort"):
        configs = []

        # =============================================================
        # jnp.sort - single tensor sort
        # Gradients enabled - uses scatter with batching dims
        # =============================================================

        # 1D sort
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x),
                lambda key: random.normal(key, (16,)),
                name="jnp.sort.1d",
            )
        )

        # 2D sort - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=0),
                lambda key: random.normal(key, (4, 8)),
                name="jnp.sort.2d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=1),
                lambda key: random.normal(key, (4, 8)),
                name="jnp.sort.2d.axis1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=-1),
                lambda key: random.normal(key, (4, 8)),
                name="jnp.sort.2d.axis_neg1",
            )
        )

        # 3D sort - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=0),
                lambda key: random.normal(key, (2, 4, 8)),
                name="jnp.sort.3d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=1),
                lambda key: random.normal(key, (2, 4, 8)),
                name="jnp.sort.3d.axis1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=2),
                lambda key: random.normal(key, (2, 4, 8)),
                name="jnp.sort.3d.axis2",
            )
        )

        # Descending sort
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, axis=-1, descending=True),
                lambda key: random.normal(key, (4, 8)),
                name="jnp.sort.descending",
            )
        )

        # =============================================================
        # lax.sort_key_val - sort with associated values
        # Gradients enabled for both keys and values
        # =============================================================

        # 2D key-value sort on axis 0
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=0),
                lambda key: random.normal(key, (4, 8)),
                lambda key: random.normal(key, (4, 8)),
                name="lax.sort_key_val.2d.axis0",
            )
        )

        # 2D key-value sort on axis 1
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=1),
                lambda key: random.normal(key, (4, 8)),
                lambda key: random.normal(key, (4, 8)),
                name="lax.sort_key_val.2d.axis1",
            )
        )

        # 3D key-value sort
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=0),
                lambda key: random.normal(key, (2, 4, 8)),
                lambda key: random.normal(key, (2, 4, 8)),
                name="lax.sort_key_val.3d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=2),
                lambda key: random.normal(key, (2, 4, 8)),
                lambda key: random.normal(key, (2, 4, 8)),
                name="lax.sort_key_val.3d.axis2",
            )
        )

        # Key-value with stable sort
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(
                    keys, vals, dimension=1, is_stable=True
                ),
                lambda key: random.normal(key, (4, 8)),
                lambda key: jnp.arange(32).reshape(4, 8).astype(jnp.float32),
                name="lax.sort_key_val.stable",
            )
        )

        # =============================================================
        # lax.top_k - top k elements and indices
        # Returns (values, indices) - values are differentiable
        # =============================================================

        # 1D top_k
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                lambda key: random.normal(key, (16,)),
                name="lax.top_k.1d",
            )
        )

        # 2D top_k - different k values
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 1),
                lambda key: random.normal(key, (4, 8)),
                name="lax.top_k.2d.k1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                lambda key: random.normal(key, (4, 8)),
                name="lax.top_k.2d.k3",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 5),
                lambda key: random.normal(key, (4, 8)),
                name="lax.top_k.2d.k5",
            )
        )

        # 3D top_k
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 2),
                lambda key: random.normal(key, (2, 4, 8)),
                name="lax.top_k.3d",
            )
        )

        # 4D top_k
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                lambda key: random.normal(key, (2, 2, 4, 8)),
                name="lax.top_k.4d",
            )
        )

        # top_k after transpose (tests non-last-axis behavior)
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(jnp.swapaxes(x, 0, 2), 2),
                lambda key: random.normal(key, (2, 4, 8)),
                name="lax.top_k.transposed",
            )
        )

        # top_k with k equal to axis size
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 5),
                lambda key: random.normal(key, (4, 8)),
                name="lax.top_k.k_equals_size",
            )
        )

        # =============================================================
        # jnp.argmax - argmax (multi-result reduce with max)
        # =============================================================

        # 1D argmax
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x),
                lambda key: random.normal(key, (16,)),
                differentiable_argnums=(),
                name="jnp.argmax.1d",
            )
        )

        # 2D argmax - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=0),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.2d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=1),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.2d.axis1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=-1),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.2d.axis_neg1",
            )
        )

        # 3D argmax
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=0),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.3d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=1),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.3d.axis1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=2),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.3d.axis2",
            )
        )

        # argmax with keepdims
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmax(x, axis=1, keepdims=True),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmax.keepdims",
            )
        )

        # =============================================================
        # jnp.argmin - argmin (multi-result reduce with min)
        # =============================================================

        # 1D argmin
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x),
                lambda key: random.normal(key, (16,)),
                differentiable_argnums=(),
                name="jnp.argmin.1d",
            )
        )

        # 2D argmin - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=0),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.2d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=1),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.2d.axis1",
            )
        )

        # 3D argmin
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=0),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.3d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=2),
                lambda key: random.normal(key, (2, 4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.3d.axis2",
            )
        )

        # argmin with keepdims
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argmin(x, axis=0, keepdims=True),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argmin.keepdims",
            )
        )

        # =============================================================
        # jnp.argsort - indices that would sort
        # =============================================================

        # 1D argsort
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x),
                lambda key: random.normal(key, (16,)),
                differentiable_argnums=(),
                name="jnp.argsort.1d",
            )
        )

        # 2D argsort - different axes
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x, axis=0),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argsort.2d.axis0",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x, axis=1),
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="jnp.argsort.2d.axis1",
            )
        )

        # argsort descending
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x, descending=True),
                lambda key: random.normal(key, (16,)),
                differentiable_argnums=(),
                name="jnp.argsort.descending",
            )
        )

        # top_k with ties: test stable tie ordering (negate approach)
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                jnp.array([3.0, 1.0, 3.0, 2.0, 3.0]),
                name="lax.top_k.ties",
            )
        )

        # top_k k=1 with ties: exercises the argmax fast path, must return
        # the lowest original index among tied maxima.
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 1),
                jnp.array([3.0, 1.0, 3.0, 2.0, 3.0]),
                name="lax.top_k.ties.k1",
            )
        )

        # top_k with integer input (exercises bitwise-NOT descending key)
        configs.append(
            OperationTestConfig(
                lambda x: lax.top_k(x, 3),
                jnp.array([5, 3, 8, 1, 7], dtype=jnp.int32),
                differentiable_argnums=(),
                name="lax.top_k.int32",
            )
        )

        # searchsorted with NaN (exercises TOTALORDER comparison)
        configs.append(
            OperationTestConfig(
                lambda x: jnp.searchsorted(x, x),
                jnp.array([-jnp.inf, -1.0, 0.0, 1.0, jnp.inf, jnp.nan]),
                differentiable_argnums=(),
                name="jnp.searchsorted.nan",
            )
        )

        # =============================================================
        # bool sort — MLX's Metal block_sort kernel has no bool variant
        # (Unable to load kernel ncarg_block_sort_bool__uint32_*). The
        # handler casts bool → int8 around the sort. Exercised by
        # jnp.compress / jnp.extract via lax.sort, plus argsort/lexsort.
        # =============================================================
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x),
                jnp.array([True, False, True, True, False, False, True]),
                differentiable_argnums=(),
                name="jnp.sort.bool.1d",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x),
                jnp.array([[True, False, True], [False, True, False]]),
                differentiable_argnums=(),
                name="jnp.argsort.bool.2d",
            )
        )
        # sort-by-key with bool keys exercises the multi-input path
        # (argsort over a bool tensor + take_along_axis on values).
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=0),
                jnp.array([True, False, True, False, True]),
                jnp.arange(5, dtype=jnp.float32),
                differentiable_argnums=(1,),
                name="lax.sort_key_val.bool_keys",
            )
        )
        # Descending bool sort — exercises the `descending=True` branch.
        configs.append(
            OperationTestConfig(
                lambda x: jnp.sort(x, descending=True),
                jnp.array([True, False, True, True, False, False, True]),
                differentiable_argnums=(),
                name="jnp.sort.bool.descending",
            )
        )
        # Stable descending argsort with tie-heavy bool keys: a naive
        # cast→argsort ascending→reverse would invert tie order within the
        # equal-True and equal-False groups; the handler sorts on a flipped
        # key to preserve stable descending semantics.
        configs.append(
            OperationTestConfig(
                lambda x: jnp.argsort(x, descending=True, stable=True),
                jnp.array([True, False, True, False, True]),
                differentiable_argnums=(),
                name="jnp.argsort.bool.descending_stable",
            )
        )
        # sort_key_val with bool values + numeric keys — exercises the
        # take_along_axis-on-bool branch where bool values pass through
        # without casting.
        configs.append(
            OperationTestConfig(
                lambda keys, vals: lax.sort_key_val(keys, vals, dimension=0),
                jnp.arange(5, dtype=jnp.float32),
                jnp.array([True, False, True, False, True]),
                differentiable_argnums=(0,),
                name="lax.sort_key_val.bool_values",
            )
        )

        # =============================================================
        # lax.approx_max_k / approx_min_k — approximate top-k.
        # On MPS these go through a custom platform lowering that does a
        # recall-driven strided reduction for large inputs and falls back to
        # exact top-k otherwise. These configs deliberately target the
        # exact-equivalent settings (k=1, or N small enough to fit one
        # reduction tile so no reduction happens) so results match CPU exactly;
        # the genuinely-approximate path's recall is validated by the upstream
        # ann_test.py suite. Inputs use random normals (no ties) so the
        # exact-path index selection is unambiguous.
        # =============================================================

        # k=1 exact fast path (argmax-equivalent), max and min directions.
        # Gradients enabled (values output is differentiable, like lax.top_k).
        configs.append(
            OperationTestConfig(
                lambda x: lax.approx_max_k(x, 1),
                lambda key: random.normal(key, (4, 32)),
                name="lax.approx_max_k.k1",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: lax.approx_min_k(x, 1),
                lambda key: random.normal(key, (4, 32)),
                name="lax.approx_min_k.k1",
            )
        )

        # Small N, k>1: fits a single reduction tile -> no reduction -> exact.
        configs.append(
            OperationTestConfig(
                lambda x: lax.approx_max_k(x, 3),
                lambda key: random.normal(key, (4, 16)),
                name="lax.approx_max_k.small_n.k3",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: lax.approx_min_k(x, 3),
                lambda key: random.normal(key, (4, 16)),
                name="lax.approx_min_k.small_n.k3",
            )
        )

        # Non-last reduction_dimension: exercises the moveaxis path.
        configs.append(
            OperationTestConfig(
                lambda x: lax.approx_max_k(x, 1, reduction_dimension=0),
                lambda key: random.normal(key, (8, 4)),
                name="lax.approx_max_k.axis0",
            )
        )

        # Approximate path (N over the strided-reduction crossover, k>1). A
        # strictly monotonic row makes the approximation exact by construction:
        # the top-k all fall in distinct reduction slots, so none is evicted by
        # the block-max. This exercises the real strided reduce / block-argmax /
        # gather / candidate-top-k code (not the exact fallback) while still
        # matching CPU, so it fits the equality harness. (General-input recall is
        # validated by the upstream ann_test.py suite.)
        configs.append(
            OperationTestConfig(
                lambda x: lax.approx_max_k(x, 4),
                lambda key: jnp.broadcast_to(
                    jnp.arange(20000, 0, -1, dtype=jnp.float32), (2, 20000)
                ),
                name="lax.approx_max_k.reduced",
            )
        )
        configs.append(
            OperationTestConfig(
                lambda x: lax.approx_min_k(x, 4),
                lambda key: jnp.broadcast_to(
                    jnp.arange(20000, dtype=jnp.float32), (2, 20000)
                ),
                name="lax.approx_min_k.reduced",
            )
        )
        # aggregate_to_topk=False: returns the full reduced candidate set; checks
        # the non-aggregating shape/order path against CPU.
        configs.append(
            OperationTestConfig(
                lambda x: lax.approx_max_k(x, 4, aggregate_to_topk=False),
                lambda key: jnp.broadcast_to(
                    jnp.arange(20000, 0, -1, dtype=jnp.float32), (2, 20000)
                ),
                name="lax.approx_max_k.no_aggregate",
            )
        )

        return configs
