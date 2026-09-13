"""A scan's stacked outputs must survive a mid-eval command-buffer split.

MLX registers a dynamic slice's donated offset buffer as an encoder temporary,
and ``end_encoding()`` drops temporaries before producer fences resolve, so the
slice can read a stale start across a command-buffer boundary. `lax.scan` with
stacked outputs hits it because the read- and write-side clamped-start graphs
are CSEd: stacked rows come back shifted, the final carry stays exact. Fixed by
``third_party/mlx/patches/14-...`` (ml-explore/mlx#4099).

``MLX_MAX_OPS_PER_BUFFER=2`` makes the otherwise-intermittent corruption
deterministic (20/20 without the patch, 0/40 with). MLX reads it once at init,
hence the subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys

# The carry is checked too: it stays exact while the stacked rows shift, so a
# carry-only comparison would pass through the bug.
_WORKLOAD = """
import os
os.environ.setdefault('JAX_PLATFORMS', 'mps')
import numpy as np, jax, jax.numpy as jnp
from jax import lax, random

T, K, REPEATS = 128, 4, 5

def body(carry, _):
    key, state = carry
    key, sub = random.split(key)
    state = state + random.normal(sub, state.shape)
    return (key, state), state

def scan_f(key, state, T):
    return lax.scan(body, (key, state), None, length=T)

def ref_f(key, state, T):
    carry, hist = (key, state), []
    for _ in range(T):
        carry, y = body(carry, None)
        hist.append(y)
    return carry, jnp.stack(hist)

device = jax.devices('mps')[0]
key = random.PRNGKey(0)
state = jnp.zeros((K,), jnp.float32, device=device)
(_, ref_carry), ref_ys = jax.jit(ref_f, static_argnums=2, backend='mps')(key, state, T)
ref_carry, ref_ys = np.asarray(ref_carry), np.asarray(ref_ys)

for attempt in range(REPEATS):
    (_, carry), ys = jax.jit(scan_f, static_argnums=2, backend='mps')(key, state, T)
    carry, ys = np.asarray(carry), np.asarray(ys)
    if not np.allclose(ref_ys, ys, rtol=1e-5, atol=1e-5):
        row = int(np.argmax(~np.isclose(ref_ys, ys, rtol=1e-5, atol=1e-5).all(axis=1)))
        print('FENCE:MISMATCH stacked ys differ on attempt', attempt, 'first bad row', row)
        print('  expected', ref_ys[row], 'got', ys[row])
        break
    if not np.allclose(ref_carry, carry, rtol=1e-5, atol=1e-5):
        print('FENCE:MISMATCH carry differs on attempt', attempt)
        break
else:
    print('FENCE:OK')
"""


def test_scan_stacked_outputs_survive_command_buffer_split(mps_device):
    result = subprocess.run(
        [sys.executable, "-c", _WORKLOAD],
        env={**os.environ, "JAX_PLATFORMS": "mps", "MLX_MAX_OPS_PER_BUFFER": "2"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "FENCE:OK" in result.stdout, (
        f"stacked outputs disagreed with the stepped-out reference.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
