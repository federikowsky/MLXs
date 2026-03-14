"""KV cache module — cache types and factory (§6.2, §9, AC4).

Provides:
- KVCache: Full-precision, unbounded, chunked pre-allocation.
- QuantizedKVCache: Reduced memory via quantization.
- RotatingKVCache: Bounded size with circular rotation.
- ArraysCache: State arrays for SSM/recurrent models (Mamba, etc.).
- CacheList: Wrapper of multiple caches per layer (Falcon-H1, Jamba).
"""

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.cache_list import CacheList
from mlxs.cache.chunked import ChunkedKVCache
from mlxs.cache.kv import KVCache
from mlxs.cache.quantized import QuantizedKVCache
from mlxs.cache.rotating import RotatingKVCache

__all__ = [
    "ArraysCache",
    "CacheList",
    "ChunkedKVCache",
    "KVCache",
    "QuantizedKVCache",
    "RotatingKVCache",
    "convert_to_quantized",
    "create_cache",
]


def create_cache(
    num_layers: int,
    *,
    max_kv_size: int | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    keep: int = 0,
    quantized_kv_start: int = 0,
) -> list[KVCache | QuantizedKVCache | RotatingKVCache]:
    """Create a list of KV caches for a model (factory function).

    Args:
        num_layers: Number of transformer layers.
        max_kv_size: If set, use RotatingKVCache with this max size.
        kv_bits: If set, use QuantizedKVCache with this bit width.
        kv_group_size: Group size for quantized cache.
        keep: Tokens to always keep in rotating cache.
        quantized_kv_start: When >0 and kv_bits set, start with full-precision
            caches and convert after this many decode steps (FR4).

    Returns:
        List of cache instances, one per layer.
    """
    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in range(num_layers)]
    if kv_bits is not None:
        if quantized_kv_start > 0:
            # Start full-precision; caller converts after quantized_kv_start steps
            return [KVCache() for _ in range(num_layers)]
        return [
            QuantizedKVCache(group_size=kv_group_size, bits=kv_bits) for _ in range(num_layers)
        ]
    return [KVCache() for _ in range(num_layers)]


def convert_to_quantized(
    cache: list[KVCache],
    *,
    kv_bits: int = 8,
    kv_group_size: int = 64,
) -> list[QuantizedKVCache]:
    """Convert full-precision KVCache list to QuantizedKVCache (FR4, §6.2).

    Used when ``quantized_kv_start > 0``: after N decode steps, the existing
    full-precision KV data is quantized and stored in QuantizedKVCache.

    Args:
        cache: List of full-precision KVCache instances.
        kv_bits: Quantization bit width.
        kv_group_size: Quantization group size.

    Returns:
        List of QuantizedKVCache instances with the same KV data.
    """
    import mlx.core as mx

    result: list[QuantizedKVCache] = []
    for layer in cache:
        qc = QuantizedKVCache(group_size=kv_group_size, bits=kv_bits)
        if not layer.empty() and layer.keys is not None and layer.values is not None:
            # Feed existing full-precision data into the quantized cache
            qc.update_and_fetch(layer.keys, layer.values)
            mx.eval(qc.state)
        result.append(qc)
    return result
