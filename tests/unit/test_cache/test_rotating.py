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
