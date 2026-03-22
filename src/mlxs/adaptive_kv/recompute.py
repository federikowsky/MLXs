"""Adaptive recomputation coordination."""

from __future__ import annotations

from collections.abc import Callable

from mlxs.adaptive_kv.block_registry import AdaptiveBlockRegistry
from mlxs.adaptive_kv.block_types import BlockTier, RecomputeRequest
from mlxs.adaptive_kv.exceptions import AdaptiveKVRecoveryNotImplementedError


class AdaptiveRecomputeCoordinator:
    """Tracks recovery requests for evicted blocks."""

    def __init__(
        self,
        registry: AdaptiveBlockRegistry,
        *,
        on_request: Callable[[], None] | None = None,
        on_recover: Callable[[RecomputeRequest], None] | None = None,
    ) -> None:
        self._registry = registry
        self._on_request = on_request
        self._on_recover = on_recover

    def request(self, block_ids: tuple[int, ...], *, reason: str) -> RecomputeRequest:
        spans: list[tuple[int, int]] = []
        for block_id in block_ids:
            block = self._registry.get(block_id)
            if block.tier is not BlockTier.EVICTED:
                raise ValueError(
                    f"Adaptive recomputation requires evicted blocks, got tier={block.tier.value}"
                )
            spans.append((block.source_start, block.source_end))
        if self._on_request is not None:
            self._on_request()
        return RecomputeRequest(block_ids=block_ids, source_spans=tuple(spans), reason=reason)

    def recover(self, request: RecomputeRequest) -> None:
        if self._on_recover is None:
            raise AdaptiveKVRecoveryNotImplementedError(
                "Adaptive KV replay-based recovery is not completed in this V1 patch set."
            )
        self._on_recover(request)
