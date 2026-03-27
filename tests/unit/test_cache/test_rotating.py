from __future__ import annotations

import mlx.core as mx

from mlxs.cache.rotating import RotatingKVCache


def _kv_from_tokens(tokens: list[int]) -> tuple[mx.array, mx.array]:
    base = mx.array(tokens, dtype=mx.float32).reshape(1, 1, len(tokens), 1)
    keys = mx.broadcast_to(base, (1, 1, len(tokens), 2))
    values = mx.broadcast_to(base + 100.0, (1, 1, len(tokens), 2))
    return keys, values


def test_rotating_cache_handles_decode_after_long_prefill() -> None:
    cache = RotatingKVCache(max_size=4, keep=0)

    prefill_k, prefill_v = _kv_from_tokens([1, 2, 3, 4, 5, 6])
    cache.update_and_fetch(prefill_k, prefill_v)

    decode_k, decode_v = _kv_from_tokens([7])
    keys, values = cache.update_and_fetch(decode_k, decode_v)

    assert keys.shape == (1, 1, 4, 2)
    assert values.shape == (1, 1, 4, 2)
    assert cache.offset == 7
    assert sorted(int(token) for token in keys[0, 0, :, 0].tolist()) == [4, 5, 6, 7]
    assert sorted(int(token) for token in values[0, 0, :, 0].tolist()) == [104, 105, 106, 107]


def test_rotating_cache_returns_full_first_prefill_attention_view() -> None:
    cache = RotatingKVCache(max_size=4, keep=0)

    keys, values = _kv_from_tokens([1, 2, 3, 4, 5, 6])
    out_k, out_v = cache.update_and_fetch(keys, values)

    assert out_k.shape == (1, 1, 6, 2)
    assert out_v.shape == (1, 1, 6, 2)
    assert cache.offset == 6
    assert out_k[0, 0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert out_v[0, 0, :, 0].tolist() == [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
