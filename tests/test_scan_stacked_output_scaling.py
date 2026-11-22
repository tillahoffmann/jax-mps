"""A scan's stacked-output cost must stay linear in the trip count (jax-mps#215).

`lax.scan` collects per-step outputs by `dynamic_update_slice`-ing each row into
a loop-carried (T, ...) buffer. MLX writes that row in place only when the
buffer is donatable, i.e. nothing else references it. The counted while loop
flushes its lazy graph every 64 iterations, and the command buffer of each
flush keeps the evaluated buffers referenced until its completion handler runs.
If the next chunk is encoded before that, its first `dynamic_update_slice`
copies the whole (T, ...) buffer — once per chunk, so the total cost is
quadratic in T.

Timing is the only observable signal: peak memory stays ~2x the buffer either
way. Both lengths sit in the steady regime (>= 256 MiB stacked) where per-step
cost is flat once the fix is in. Doubling T takes ~2.0x with the fix and
3.1-3.6x without it on an M-series GPU; the limits sit between the two.
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

N = 1 << 16  # 256 KiB float32 rows: 256 MiB stacked at T=1024, 512 MiB at T=2048.


def _stacked_scan(T: int):
    def step(c, _):
        c = jnp.tanh(0.99 * c + 0.01)
        return c, c

    return jax.jit(lambda x: jax.lax.scan(step, x, None, length=T)[1][-1])


def _median_time(f, x, repeats: int = 5) -> float:
    jax.block_until_ready(f(x))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(f(x))
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def test_stacked_scan_cost_is_linear_in_length(mps_device):
    x = jax.device_put(jnp.linspace(-1.0, 1.0, N), mps_device)
    short = _median_time(_stacked_scan(1024), x)
    long = _median_time(_stacked_scan(2048), x)
    ratio = long / short
    assert ratio < 2.6, (
        f"stacked scan T=2048 took {long * 1e3:.1f} ms vs {short * 1e3:.1f} ms at "
        f"T=1024 ({ratio:.2f}x for 2x the steps); expected ~linear scaling"
    )


def _stacked_while(T: int, n: int):
    # A data-dependent condition keeps this off the counted fast path, so each
    # iteration is evaluated on its own. The previous iteration's buffer must
    # not be held across that eval, or every dynamic_update_slice copies the
    # whole (T, n) buffer.
    def f(x):
        def cond(state):
            i, c, _ = state
            return jnp.logical_and(i < T, c[0] < 1e30)

        def body(state):
            i, c, buf = state
            c = jnp.tanh(0.99 * c + 0.01)
            return i + 1, c, jax.lax.dynamic_update_slice(buf, c[None], (i, 0))

        init = (0, x, jnp.zeros((T, n), x.dtype))
        return jax.lax.while_loop(cond, body, init)[2][-1]

    return jax.jit(f)


def test_stacked_while_loop_cost_is_linear_in_length(mps_device):
    n = 1 << 17  # 512 KiB rows: 64 MiB stacked at T=128, 128 MiB at T=256.
    x = jax.device_put(jnp.linspace(-1.0, 1.0, n), mps_device)
    short = _median_time(_stacked_while(128, n), x)
    long = _median_time(_stacked_while(256, n), x)
    ratio = long / short
    assert ratio < 2.7, (
        f"stacked while_loop T=256 took {long * 1e3:.1f} ms vs {short * 1e3:.1f} ms "
        f"at T=128 ({ratio:.2f}x for 2x the steps); expected ~linear scaling"
    )
