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
from mlxs.cache.kv import KVCache
from mlxs.cache.quantized import QuantizedKVCache
from mlxs.cache.rotating import RotatingKVCache

__all__ = [
    "ArraysCache",
    "CacheList",
    "KVCache",
    "QuantizedKVCache",
    "RotatingKVCache",
    "create_cache",
]


def create_cache(
    num_layers: int,
    *,
    max_kv_size: int | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    keep: int = 0,
) -> list[KVCache | QuantizedKVCache | RotatingKVCache]:
    """Create a list of KV caches for a model (factory function).

    Args:
        num_layers: Number of transformer layers.
        max_kv_size: If set, use RotatingKVCache with this max size.
        kv_bits: If set, use QuantizedKVCache with this bit width.
        kv_group_size: Group size for quantized cache.
        keep: Tokens to always keep in rotating cache.

    Returns:
        List of cache instances, one per layer.
    """
    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in range(num_layers)]
    if kv_bits is not None:
        return [
            QuantizedKVCache(group_size=kv_group_size, bits=kv_bits) for _ in range(num_layers)
        ]
    return [KVCache() for _ in range(num_layers)]
