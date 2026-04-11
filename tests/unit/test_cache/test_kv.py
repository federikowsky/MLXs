"""Tests for KV cache growth behavior."""

from __future__ import annotations

import mlx.core as mx

from mlxs.cache.kv import KVCache


def test_initial_prefill_boundary_reserves_one_extra_chunk(monkeypatch) -> None:
    monkeypatch.setattr(KVCache, "step", 4)

    cache = KVCache()
    keys = mx.zeros((1, 1, 3, 2))
    values = mx.zeros((1, 1, 3, 2))
    cache.update_and_fetch(keys, values)

    assert cache._keys is not None
    assert cache._keys.shape == (1, 1, 8, 2)
    assert cache._offset == 3

    cache.update_and_fetch(mx.zeros((1, 1, 1, 2)), mx.zeros((1, 1, 1, 2)))
    assert cache._keys.shape == (1, 1, 8, 2)
    assert cache._offset == 4
