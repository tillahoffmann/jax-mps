import numpy
from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def make_control_flow_op_configs():
    with OperationTestConfig.module_name("control_flow"):
        return [
            # ==================== lax.cond (2-branch case) ====================
            OperationTestConfig(
                lambda pred, x, y: lax.cond(pred, lambda: x + 1, lambda: y * 2),
                numpy.bool_(True),
                numpy.float32(3.0),
                numpy.float32(4.0),
                differentiable_argnums=(),
                name="lax.cond.true",
            ),
            OperationTestConfig(
                lambda pred, x, y: lax.cond(pred, lambda: x + 1, lambda: y * 2),
                numpy.bool_(False),
                numpy.float32(3.0),
                numpy.float32(4.0),
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
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
                lambda key: random.normal(key, (4, 8)),
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
                lambda key: random.normal(key, (4, 8)),
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
                differentiable_argnums=(),
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
                differentiable_argnums=(),
                name="lax.while_loop.with_switch",
            ),
        ]
