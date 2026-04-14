import jax
import numpy
from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def _make_lstm_params(key, input_dim, hidden_dim):
    keys = random.split(key, 8)
    # Wi, Wf, Wc, Wo (input projections) + Ui, Uf, Uc, Uo (recurrent projections)
    return tuple(
        random.normal(k, (input_dim if i < 4 else hidden_dim, hidden_dim)) * 0.01
        for i, k in enumerate(keys)
    )


def _lstm_step(carry, x):
    (h, c), params = carry
    Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo = params
    i = jax.nn.sigmoid(x @ Wi + h @ Ui)
    f = jax.nn.sigmoid(x @ Wf + h @ Uf)
    g = jnp.tanh(x @ Wc + h @ Uc)
    o = jax.nn.sigmoid(x @ Wo + h @ Uo)
    c_new = f * c + i * g
    h_new = o * jnp.tanh(c_new)
    return ((h_new, c_new), params), h_new


def _scan_lstm_loss(params):
    """Loss function from issue #134 reproducer. Crashes with metal::malloc on main."""
    hidden = 64
    xs = jnp.ones((10_000, 32))
    init = ((jnp.zeros(hidden), jnp.zeros(hidden)), params)
    _, ys = jax.lax.scan(_lstm_step, init, xs)
    return ys.sum()


def _inner_fn_with_scan(x):
    """Simulates _preprocess from jax-baseline: uses lax.scan (→ stablehlo.while)."""

    def scan_body(carry, elem):
        return carry + elem, carry + elem

    _, ys = lax.scan(scan_body, jnp.float32(0.0), x)
    return ys


def _func_call_with_scan(x):
    """Calls _inner_fn_with_scan via jax.jit, generating a func.call in StableHLO."""
    return jax.jit(_inner_fn_with_scan)(x).sum()


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
            # ==================== lax.scan regression (issue #134) ====================
            # LSTM recurrence with grad: exercises the WhileLoopPrimitive
            # graph-bounding path. 10_000 timesteps triggers graph explosion
            # on main (metal::malloc crash). jax.grad is baked into the tested
            # function so the value test exercises the backward pass (the crash
            # path). In compare mode, the harness still does a strict allclose
            # on the returned scalar; this config is primarily a regression
            # smoke test for completing the backward pass without crashing.
            OperationTestConfig(
                lambda params: jax.grad(_scan_lstm_loss)(params)[0].sum(),
                lambda key: _make_lstm_params(key, 32, 64),
                name="lax.scan.lstm_grad",
            ),
        ]
