"""Tests for buffer donation (``jax.jit(donate_argnums=...)``) on MPS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Large enough that a retained 64 MiB input is unambiguous against the noise of
# whatever else the process has allocated.
_MIB = 1024 * 1024
_BIG_BYTES = 64 * _MIB
_BIG_N = _BIG_BYTES // 4  # float32


def _on_device(array, device):
    out = jax.device_put(array, device)
    out.block_until_ready()
    return out


def test_donation_aliasing_output(mps_device):
    """
    Test invalidation for `donate_argnums` with identical shape and dtype.

    This goes through the `tf.aliasing_output` property in StableHLO.
    """

    add = jax.jit(lambda a, b: a + b, donate_argnums=0)

    a = _on_device(np.arange(5, dtype=np.float32), mps_device)
    b = _on_device(np.ones(5, dtype=np.float32), mps_device)

    out = add(a, b)

    np.testing.assert_allclose(np.asarray(out), np.arange(5) + 1)
    assert a.is_deleted(), "donated argument should be invalidated"
    assert not b.is_deleted(), "non-donated argument should stay usable"

    # reading an invalidated array should error
    with pytest.raises(RuntimeError):
        np.asarray(a)


def test_donation_buffer_donor(mps_device):
    """
    Test invalidation for `donate_argnums` with identical size but not shape.

    This goes through the `jax.buffer_donor` property in StableHLO.
    """

    add = jax.jit(lambda a, b: (a + b).reshape(6), donate_argnums=0)

    a = _on_device(np.ones((2, 3), dtype=np.float32), mps_device)
    b = _on_device(np.ones((2, 3), dtype=np.float32), mps_device)

    out = add(a, b)

    np.testing.assert_allclose(np.asarray(out), np.full(6, 2.0, dtype=np.float32))
    assert a.is_deleted(), "donated argument should be invalidated"
    assert not b.is_deleted(), "non-donated argument should stay usable"

    # reading an invalidated array should error
    with pytest.raises(RuntimeError):
        np.asarray(a)


def test_donation_multiple_consumers(mps_device):
    """Reclaiming an input feeding several ops must not corrupt the result."""

    def multi_use(p):
        return p * 2.0 + p * 3.0 + jnp.sin(p)

    fn = jax.jit(multi_use, donate_argnums=0)

    host = np.arange(1, 6, dtype=np.float32)
    p = _on_device(host, mps_device)

    out = fn(p)

    np.testing.assert_allclose(
        np.asarray(out), host * 2.0 + host * 3.0 + np.sin(host), rtol=1e-6
    )
    assert p.is_deleted(), "donated argument should be invalidated"


def test_non_donatable_input_indices(mps_device):
    """Passing one array to both a donated and a non-donated parameter.

    The buffer is still owned through the non-donated position,
    so reclaiming it would be a use-after-delete.
    """
    fn = jax.jit(lambda u, v: u + v, donate_argnums=0)

    s = _on_device(np.arange(4, dtype=np.float32), mps_device)

    out = fn(s, s)

    np.testing.assert_allclose(np.asarray(out), np.arange(4) * 2)


def test_donation_saves_memory(mps_device):
    """Test that buffer donation actually saves memory"""

    def chain(v):
        return (v * 2.0 + 1.0) * 0.5

    def live_bytes():
        return mps_device.memory_stats()["bytes_in_use"]

    def delta(fn):
        x = _on_device(np.ones(_BIG_N, dtype=np.float32), mps_device)
        before = live_bytes()
        # `out` and `x` must both still be referenced at the measurement, or
        # the output would be collected and its bytes freed before we look.
        out = fn(x)
        out.block_until_ready()
        grew = live_bytes() - before
        del out, x
        return grew

    donated_delta = delta(jax.jit(chain, donate_argnums=0))
    kept_delta = delta(jax.jit(chain))

    assert kept_delta >= 0.9 * _BIG_BYTES, (
        f"expected the non-donating call to retain the input "
        f"(~{_BIG_BYTES} bytes), saw {kept_delta}"
    )
    assert donated_delta < 0.5 * _BIG_BYTES, (
        f"expected donation to reclaim the input, but live memory grew by "
        f"{donated_delta} bytes (non-donating call grew by {kept_delta})"
    )
