from __future__ import annotations

import mlx.core as mx

from mlxs.cache.kv import KVCache
from mlxs.cache.quantized import QuantizedKVCache


def _kv_arrays(length: int) -> tuple[mx.array, mx.array]:
    values = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
    return values, values + 100.0


def _quantized_arrays(length: int) -> tuple[mx.array, mx.array]:
    base = mx.arange(length * 32, dtype=mx.float32).reshape(1, 1, length, 32)
    return base, base + 100.0


def test_kvcache_remove_middle_range_preserves_order() -> None:
    keys, values = _kv_arrays(5)
    cache = KVCache()
    cache.update_and_fetch(keys, values)

    removed = cache.remove_token_range(1, 3)

    assert removed == 2
    assert cache.offset == 3
    assert cache.keys is not None
    assert cache.keys.shape[2] == 3
    mx.eval(cache.keys)
    assert [float(v.item()) for v in cache.keys[0, 0, :, 0]] == [0.0, 3.0, 4.0]


def test_quantized_cache_remove_middle_range_preserves_length() -> None:
    keys, values = _quantized_arrays(5)
    cache = QuantizedKVCache(bits=8, group_size=32)
    cache.update_and_fetch(keys, values)

    removed = cache.remove_token_range(2, 4)

    assert removed == 2
    assert cache.offset == 3
    assert cache.keys is not None
    assert cache.keys[0].shape[2] == 3


def test_quantized_state_setter_updates_offset() -> None:
    keys, values = _quantized_arrays(4)
    cache = QuantizedKVCache(bits=8, group_size=32)
    cache.update_and_fetch(keys, values)
    state = cache.state
    other = QuantizedKVCache(bits=8, group_size=32)
    other.state = state  # type: ignore[assignment]

    assert other.offset == 4


def test_kvcache_live_state_size_bytes_tracks_only_live_prefix() -> None:
    keys, values = _quantized_arrays(2)
    cache = KVCache()
    cache.update_and_fetch(keys, values)

    assert cache.live_state_size_bytes == keys.nbytes + values.nbytes
    assert cache.state_size_bytes > cache.live_state_size_bytes


def test_quantized_cache_live_state_size_bytes_tracks_only_live_prefix() -> None:
    keys, values = _quantized_arrays(2)
    cache = QuantizedKVCache(bits=8, group_size=32)
    cache.update_and_fetch(keys, values)

    assert cache.live_state_size_bytes > 0
    assert cache.state_size_bytes > cache.live_state_size_bytes
