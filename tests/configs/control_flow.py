import jax
import numpy
from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def _inner_fn_with_scan(x):
    """Simulates _preprocess from jax-baseline: uses lax.scan (→ stablehlo.while)."""

    def scan_body(carry, elem):
        return carry + elem, carry + elem

    _, ys = lax.scan(scan_body, jnp.float32(0.0), x)
    return ys


def _func_call_with_scan(x):
    """Calls _inner_fn_with_scan via jax.jit, generating a func.call in StableHLO."""
    return jax.jit(_inner_fn_with_scan)(x).sum()


def _long_scan_grad(xs, alpha):
    """Long scan + grad — regression for issue #134 (unbounded graph growth on
    reverse-mode AD through lax.scan caused metal::malloc to exhaust at large
    iteration counts). WhileLoopPrimitive bounds graph depth via per-iter eval.
    """

    def body(carry, x_t):
        new = carry * jnp.float32(0.99) + x_t * alpha
        return new, new

    _, ys = lax.scan(body, jnp.float32(0.0), xs)
    return ys.sum()


@jax.checkpoint
def _checkpointed_fn(x):
    """Checkpointed function that generates optimization_barrier in StableHLO."""
    return jnp.tanh(x * 2 + 1)


def make_control_flow_op_configs():
    with OperationTestConfig.module_name("control_flow"):
        return [
            # ==================== func.call with scan (issue #91) ====================
            OperationTestConfig(
                _func_call_with_scan,
                lambda key: random.normal(key, (8,)),
                name="func_call.scan",
            ),
            # ==================== long scan + grad (issue #134) ====================
            OperationTestConfig(
                _long_scan_grad,
                lambda key: random.normal(key, (2000,)),
                numpy.float32(0.5),
                name="lax.scan.long_grad",
            ),
            # ==================== jax.checkpoint / optimization_barrier (issue #91) ====================
            OperationTestConfig(
                lambda x: _checkpointed_fn(x).sum(),
                lambda key: random.normal(key, (4, 4)),
                name="checkpoint.forward",
            ),
            # ==================== lax.cond (2-branch case) ====================
            OperationTestConfig(
                lambda pred, x, y: lax.cond(pred, lambda: x + 1, lambda: y * 2),
                numpy.bool_(True),
                numpy.float32(3.0),
                numpy.float32(4.0),
                name="lax.cond.true",
            ),
            OperationTestConfig(
                lambda pred, x, y: lax.cond(pred, lambda: x + 1, lambda: y * 2),
                numpy.bool_(False),
                numpy.float32(3.0),
                numpy.float32(4.0),
                name="lax.cond.false",
            ),
            OperationTestConfig(
                lambda pred, x: lax.cond(
                    pred,
                    lambda a: a + jnp.flip(a),
                    lambda a: a * 2,
                    x,
                ),
                numpy.bool_(True),
                lambda key: random.normal(key, (4,)),
                name="lax.cond.array",
            ),
            # ==================== lax.switch ====================
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [lambda y: y + 1, lambda y: y * 2, lambda y: y - 3],
                    x,
                ),
                numpy.int32(1),
                lambda key: random.normal(key, (4,)),
                name="lax.switch",
            ),
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [
                        lambda y: y + jnp.flip(y, axis=0),
                        lambda y: y + jnp.flip(y, axis=1),
                        lambda y: y + jnp.swapaxes(y, 0, 1),
                    ],
                    x,
                ),
                numpy.int32(2),
                lambda key: random.normal(key, (4, 4)),
                name="lax.switch.multiaxis",
            ),
            # Boundary selector: first branch (index 0)
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [lambda y: y + 1, lambda y: y * 2, lambda y: y - 3],
                    x,
                ),
                numpy.int32(0),
                numpy.float32(5.0),
                name="lax.switch.first_branch",
            ),
            # Boundary selector: last branch (index N-1)
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [lambda y: y + 1, lambda y: y * 2, lambda y: y - 3],
                    x,
                ),
                numpy.int32(2),
                numpy.float32(5.0),
                name="lax.switch.last_branch",
            ),
            # Out-of-bounds selector: negative (should clamp to first or last)
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [lambda y: y + 1, lambda y: y * 2, lambda y: y - 3],
                    x,
                ),
                numpy.int32(-1),
                numpy.float32(5.0),
                name="lax.switch.oob_negative",
            ),
            # Out-of-bounds selector: too large (should clamp to last)
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [lambda y: y + 1, lambda y: y * 2, lambda y: y - 3],
                    x,
                ),
                numpy.int32(100),
                numpy.float32(5.0),
                name="lax.switch.oob_large",
            ),
            # Many branches (5 branches)
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [
                        lambda y: y + 1,
                        lambda y: y * 2,
                        lambda y: y - 3,
                        lambda y: y / 2,
                        lambda y: y**2,
                    ],
                    x,
                ),
                numpy.int32(3),
                numpy.float32(4.0),
                name="lax.switch.many_branches",
            ),
            # Multiple return values from branches
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [
                        lambda y: (y + 1, y * 2),
                        lambda y: (y - 1, y / 2),
                        lambda y: (y * 3, y + 4),
                    ],
                    x,
                )[0],
                numpy.int32(1),
                numpy.float32(5.0),
                name="lax.switch.multi_return.first",
            ),
            OperationTestConfig(
                lambda selector, x: lax.switch(
                    selector,
                    [
                        lambda y: (y + 1, y * 2),
                        lambda y: (y - 1, y / 2),
                        lambda y: (y * 3, y + 4),
                    ],
                    x,
                )[1],
                numpy.int32(1),
                numpy.float32(5.0),
                name="lax.switch.multi_return.second",
            ),
            # Nested switch inside branches
            OperationTestConfig(
                lambda outer, inner, x: lax.switch(
                    outer,
                    [
                        lambda y: lax.switch(
                            inner,
                            [lambda z: z + 1, lambda z: z + 2],
                            y,
                        ),
                        lambda y: lax.switch(
                            inner,
                            [lambda z: z * 2, lambda z: z * 3],
                            y,
                        ),
                    ],
                    x,
                ),
                numpy.int32(0),
                numpy.int32(1),
                numpy.float32(5.0),
                name="lax.switch.nested",
            ),
            # ==================== lax.while_loop ====================
            # Basic scalar accumulation
            OperationTestConfig(
                lambda init: lax.while_loop(
                    lambda state: state[0] < 5,
                    lambda state: (state[0] + 1, state[1] + state[0]),
                    (init, init),
                )[1],
                numpy.int32(0),
                differentiable_argnums=(),
                name="lax.while_loop",
            ),
            # Zero iterations (condition immediately false)
            OperationTestConfig(
                lambda init: lax.while_loop(
                    lambda state: state[0] < 0,  # Always false
                    lambda state: (state[0] + 1, state[1] * 2),
                    (init, init + 10),
                )[1],
                numpy.int32(5),
                differentiable_argnums=(),
                name="lax.while_loop.zero_iter",
            ),
            # Single iteration
            OperationTestConfig(
                lambda init: lax.while_loop(
                    lambda state: state[0] < 1,  # True once, then false
                    lambda state: (state[0] + 1, state[1] * 3),
                    (init, numpy.float32(2.0)),
                )[1],
                numpy.int32(0),
                differentiable_argnums=(),
                name="lax.while_loop.one_iter",
            ),
            # Many live carries: the compiled body+cond function emits fused
            # elementwise segments with ~30 outputs. MLX must split these to
            # respect Metal's 31-argument-buffer kernel limit (vendored MLX
            # patch 13); before that guard, fusion produced an unencodable
            # kernel whose eval-time throw on a stream worker thread was
            # swallowed, silently leaving the loop outputs as never-written
            # buffers (the numpyro.Binomial first-call corruption).
            OperationTestConfig(
                lambda init: sum(
                    lax.while_loop(
                        lambda state: state[0] < 3,
                        lambda state: (
                            (state[0] + 1,)
                            + tuple(
                                x + (state[1] * 0.5 + 1.0) * (k + 1)
                                for k, x in enumerate(state[1:])
                            )
                        ),
                        (numpy.int32(0),) + tuple(init + k for k in range(28)),
                    )[1:]
                ),
                numpy.linspace(0.0, 1.0, 4, dtype=numpy.float32),
                differentiable_argnums=(),
                name="lax.while_loop.many_carries",
            ),
            # Array operations along axis 1
            OperationTestConfig(
                lambda init: lax.while_loop(
                    lambda state: state[0] < 3,
                    lambda state: (
                        state[0] + 1,
                        state[1] + jnp.sum(state[1], axis=1, keepdims=True),
                    ),
                    (numpy.int32(0), init),
                )[1],
                # Use deterministic numpy input so both platforms start from identical data
                # (random generation on different devices can produce slightly different
                # float32 values, and 3 iterations of cumulative sum compounds the error).
                numpy.random.default_rng(42)
                .standard_normal((4, 8))
                .astype(numpy.float32),
                differentiable_argnums=(),
                name="lax.while_loop.axis1",
            ),
            # Array operations along axis 0
            OperationTestConfig(
                lambda init: lax.while_loop(
                    lambda state: state[0] < 3,
                    lambda state: (
                        state[0] + 1,
                        state[1] + jnp.sum(state[1], axis=0, keepdims=True),
                    ),
                    (numpy.int32(0), init),
                )[1],
                lambda key: random.normal(key, (4, 8)),
                differentiable_argnums=(),
                name="lax.while_loop.axis0",
            ),
            # Mixed dtypes in state (int32 counter + float32 accumulator)
            OperationTestConfig(
                lambda init_f: lax.while_loop(
                    lambda state: state[0] < 4,
                    lambda state: (state[0] + 1, state[1] + 0.5),
                    (numpy.int32(0), init_f),
                )[1],
                numpy.float32(1.0),
                differentiable_argnums=(),
                name="lax.while_loop.mixed_dtype",
            ),
            # Three-element tuple state
            OperationTestConfig(
                lambda x, y: lax.while_loop(
                    lambda state: state[0] < 3,
                    lambda state: (state[0] + 1, state[1] + 1.0, state[2] * 2.0),
                    (numpy.int32(0), x, y),
                ),
                numpy.float32(1.0),
                numpy.float32(1.0),
                differentiable_argnums=(),
                name="lax.while_loop.triple_state",
            ),
            # Nested while loops with scalars
            OperationTestConfig(
                lambda init: lax.while_loop(
                    lambda state: state[0] < 4,
                    lambda state: (
                        state[0] + 1,
                        state[1]
                        + lax.while_loop(
                            lambda inner: inner[0] < state[0] + 1,
                            lambda inner: (inner[0] + 1, inner[1] + inner[0] + 1),
                            (numpy.int32(0), numpy.int32(0)),
                        )[1],
                    ),
                    (numpy.int32(0), init),
                )[1],
                numpy.int32(0),
                differentiable_argnums=(),
                name="lax.while_loop.nested_scalar",
            ),
            # Nested while loops with arrays
            OperationTestConfig(
                lambda init: lax.while_loop(
                    lambda state: state[0] < 3,
                    lambda state: (
                        state[0] + 1,
                        state[1]
                        + lax.while_loop(
                            lambda inner: inner[0] < 2,
                            lambda inner: (
                                inner[0] + 1,
                                inner[1] + jnp.sum(inner[1], axis=1, keepdims=True),
                            ),
                            (numpy.int32(0), state[1]),
                        )[1],
                    ),
                    (numpy.int32(0), init),
                )[1],
                numpy.random.default_rng(42)
                .standard_normal((4, 8))
                .astype(numpy.float32),
                differentiable_argnums=(),
                name="lax.while_loop.nested_axis1",
            ),
            # ==================== lax.fori_loop ====================
            OperationTestConfig(
                lambda x: lax.fori_loop(
                    0,
                    5,
                    lambda i, val: val + i,
                    x,
                ),
                numpy.float32(0.0),
                differentiable_argnums=(),
                name="lax.fori_loop.scalar",
            ),
            OperationTestConfig(
                lambda x: lax.fori_loop(
                    0,
                    3,
                    lambda i, val: val + jnp.roll(val, i),
                    x,
                ),
                lambda key: random.normal(key, (4,)),
                differentiable_argnums=(),
                name="lax.fori_loop.array",
            ),
            # fori_loop with zero iterations
            OperationTestConfig(
                lambda x: lax.fori_loop(
                    5,
                    5,  # lower == upper, no iterations
                    lambda i, val: val * 100,
                    x,
                ),
                numpy.float32(7.0),
                differentiable_argnums=(),
                name="lax.fori_loop.zero_iter",
            ),
            # fori_loop long enough to cross the counted-loop eval-batch
            # boundary in WhileLoopPrimitive (issue #193; batch size 64,
            # 130 = 2 full batches + remainder)
            OperationTestConfig(
                lambda x: lax.fori_loop(
                    0,
                    130,
                    lambda i, val: val * 0.99 + i * 0.001,
                    x,
                ),
                numpy.float32(1.0),
                name="lax.fori_loop.long",
            ),
            # Long counted loop with nested control flow in the body: the
            # periodic flush must stay synchronous (async_eval deadlocks on
            # nested control-flow primitives), and this crosses the flush
            # boundary unlike the short nested-loop configs (issue #193)
            OperationTestConfig(
                lambda x: lax.fori_loop(
                    0,
                    130,
                    lambda i, val: lax.cond(
                        i % 2 == 0,
                        lambda v: v + 0.01,
                        lambda v: v * 1.001,
                        val,
                    ),
                    x,
                ),
                numpy.float32(1.0),
                # Grads exercise the issue-#195 fix: the backward scan's
                # nested-cond branch index was read before the GPU work
                # producing it completed (control-flow primitives now
                # synchronize the GPU stream at eval entry).
                name="lax.fori_loop.long_nested_cond",
            ),
            # fori_loop with a traced (runtime) upper bound — the while cond's
            # limit is a function argument, not an IR constant (issue #193)
            OperationTestConfig(
                lambda x, n: lax.fori_loop(
                    0,
                    n,
                    lambda i, val: val + 1.0 + i * 0.5,
                    x,
                ),
                numpy.float32(0.0),
                numpy.int32(37),
                # JAX rejects reverse-mode AD through fori_loop with dynamic bounds
                differentiable_argnums=(),
                name="lax.fori_loop.traced_bound",
            ),
            # scan over stacked inputs/outputs — the loop counter drives
            # dynamic_slice/dynamic_update_slice in the body (issue #193)
            OperationTestConfig(
                lambda xs: lax.scan(
                    lambda c, x: (c + x, c * 0.5),
                    jnp.float32(0.0),
                    xs,
                ),
                lambda key: random.normal(key, (67,)),
                name="lax.scan.stacked_outputs",
            ),
            # Genuinely value-dependent condition (not a counted loop): the
            # cond variable is updated by a non-constant amount each iteration,
            # so the counted-loop fast path must NOT trigger (issue #193).
            # Integer state keeps the trip count platform-independent.
            OperationTestConfig(
                lambda x: lax.while_loop(
                    lambda state: state[1] < 100,
                    lambda state: (state[0] + 1, state[1] + state[0]),
                    (x, x),
                )[1],
                numpy.int32(1),
                differentiable_argnums=(),
                name="lax.while_loop.value_dependent",
            ),
            # ==================== cond inside while ====================
            OperationTestConfig(
                lambda x: lax.while_loop(
                    lambda state: state[0] < 4,
                    lambda state: (
                        state[0] + 1,
                        lax.cond(
                            state[0] % 2 == 0,
                            lambda v: v + 1,
                            lambda v: v * 2,
                            state[1],
                        ),
                    ),
                    (numpy.int32(0), x),
                )[1],
                numpy.float32(1.0),
                differentiable_argnums=(),  # while_loop doesn't support reverse-mode AD
                name="lax.while_loop.with_cond",
            ),
            # ==================== switch inside while ====================
            OperationTestConfig(
                lambda x: lax.while_loop(
                    lambda state: state[0] < 3,
                    lambda state: (
                        state[0] + 1,
                        lax.switch(
                            state[0] % 3,
                            [
                                lambda v: v + 1,
                                lambda v: v * 2,
                                lambda v: v - 0.5,
                            ],
                            state[1],
                        ),
                    ),
                    (numpy.int32(0), x),
                )[1],
                numpy.float32(1.0),
                differentiable_argnums=(),  # while_loop doesn't support reverse-mode AD
                name="lax.while_loop.with_switch",
            ),
            # ==================== while + rng (issue #82) ====================
            # fori_loop with rng split + randint accumulation
            OperationTestConfig(
                lambda key: lax.fori_loop(
                    0,
                    1000,
                    lambda _, carry: (
                        jax.random.split(carry[0])[0],
                        carry[1]
                        + jax.random.randint(
                            jax.random.split(carry[0])[1],
                            shape=(),
                            minval=0,
                            maxval=2,
                        ),
                    ),
                    (key, jnp.int32(0)),
                )[1],
                lambda key: key,
                name="lax.fori_loop.rng_accumulate",
            ),
            # fori_loop with rng split + scatter update
            OperationTestConfig(
                lambda key: lax.fori_loop(
                    0,
                    1000,
                    lambda _, carry: (
                        jax.random.split(carry[0])[0],
                        carry[1]
                        .at[
                            jax.random.randint(
                                jax.random.split(carry[0])[1],
                                shape=(),
                                minval=0,
                                maxval=2,
                            )
                        ]
                        .set(1),
                    ),
                    (key, jnp.zeros(2, dtype=jnp.int8)),
                )[1],
                lambda key: key,
                name="lax.fori_loop.rng_scatter",
            ),
        ]
