"""Decode runtime plan for Decode Engine V3.1."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mlxs.generate.stop import StopCondition


@dataclass(frozen=True, slots=True)
class DecodePlan:
    """Frozen per-request decode policy."""

    stop: StopCondition
    decoder: Callable[[int], str]
    prompt_token_count: int
    clear_cache_interval: int
    quantized_kv_start: int
    kv_bits: int | None
    kv_group_size: int


__all__ = ["DecodePlan"]
