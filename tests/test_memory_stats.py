"""Tests for PJRT_Device_MemoryStats, backed by MLX's memory.h API.

JAX surfaces the PJRT memory-stats call as ``device.memory_stats()``, which
returns ``None`` when the backend reports ``UNIMPLEMENTED`` and a dict of
stats otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

try:
    MPS_DEVICE = jax.devices("mps")[0]
except (RuntimeError, IndexError):
    MPS_DEVICE = None

pytestmark = pytest.mark.skipif(MPS_DEVICE is None, reason="MPS device required")


def test_memory_stats_returns_dict():
    assert MPS_DEVICE is not None
    stats = MPS_DEVICE.memory_stats()
    # A populated dict (not None) proves we no longer return UNIMPLEMENTED.
    assert isinstance(stats, dict)
    # bytes_in_use is the one field PJRT requires every backend to report.
    assert "bytes_in_use" in stats
    assert isinstance(stats["bytes_in_use"], int)
    assert stats["bytes_in_use"] >= 0


def test_memory_stats_optional_fields_present():
    assert MPS_DEVICE is not None
    stats = MPS_DEVICE.memory_stats()
    # Fields we wire up from MLX memory.h. peak only grows, limit is positive.
    assert stats["peak_bytes_in_use"] >= stats["bytes_in_use"]
    assert stats["bytes_limit"] > 0
    # pool_bytes (active + cached) is at least the in-use bytes.
    assert stats["pool_bytes"] >= stats["bytes_in_use"]


def test_memory_stats_reflects_allocation():
    assert MPS_DEVICE is not None
    # Peak memory never shrinks, so after an on-device computation that
    # materializes a 4 MiB result the peak must be at least that large,
    # regardless of caching/freeing behavior.
    nbytes = 1024 * 1024 * 4  # 1024x1024 float32
    x = jax.device_put(jnp.ones((1024, 1024), dtype=jnp.float32), MPS_DEVICE)
    (x @ x).block_until_ready()
    stats = MPS_DEVICE.memory_stats()
    assert stats["peak_bytes_in_use"] >= nbytes


def test_memory_stats_unreported_fields_are_sentinel():
    assert MPS_DEVICE is not None
    # Fields MLX cannot report must carry JAX's "not set" sentinel (-1),
    # proving their `_is_set` flag is false rather than leaking the
    # uninitialized struct memory the caller hands us.
    stats = MPS_DEVICE.memory_stats()
    for field in (
        "num_allocs",
        "largest_alloc_size",
        "bytes_reserved",
        "peak_bytes_reserved",
        "largest_free_block_bytes",
    ):
        assert stats[field] == -1, f"{field} should be unset (-1), got {stats[field]}"
