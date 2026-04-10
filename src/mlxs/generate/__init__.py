"""Noncanonical public Layer 2 convenience alias.

`mlxs.generate.generate` remains only because repo tests still exercise this
public alias directly. The canonical Layer 2 boundary is
`mlxs.general_path.generate_single_request`.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import GenerateOptions, TokenEvent
from mlxs.cache.kv import KVCache
from mlxs.general_path import generate_single_request
from mlxs.protocols.generate import TokenizerProtocol


def generate(
    model: nn.Module,
    tokenizer: TokenizerProtocol,
    prompt: str | list[int],
    options: GenerateOptions | None = None,
    *,
    cache: list[KVCache] | None = None,
    input_embeddings: mx.array | None = None,
    prefill_step_size: int = 2048,
    compile_decode: bool = False,
    clear_cache_interval: int = 256,
    quantized_kv_start: int = 0,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    final_cache_out: list[list[KVCache]] | None = None,
) -> Iterator[TokenEvent]:
    """Public convenience alias over the canonical Layer 2 single-request path."""
    del quantized_kv_start, kv_bits, kv_group_size
    return generate_single_request(
        model,
        tokenizer,
        prompt,
        options,
        cache=cache,
        input_embeddings=input_embeddings,
        prefill_step_size=prefill_step_size,
        compile_decode=compile_decode,
        clear_cache_interval=clear_cache_interval,
        final_cache_out=final_cache_out,
    )


__all__ = ["generate"]
