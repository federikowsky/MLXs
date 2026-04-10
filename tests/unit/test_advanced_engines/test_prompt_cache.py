"""Tests for Layer 3 prompt-cache orchestration."""

from __future__ import annotations

from typing import Any

from mlxs.advanced_engines.prompt_cache import PromptCacheOrchestrator, PromptCachePlan


class _FakePromptCache:
    def __init__(self, *, hit: tuple[list[Any] | None, int] = (None, 0)) -> None:
        self.hit = hit
        self.put_calls: list[tuple[str, tuple[int, ...], list[Any]]] = []

    def get(self, model_id: str, token_ids: tuple[int, ...]) -> tuple[list[Any] | None, int]:
        self.last_get = (model_id, token_ids)
        return self.hit

    def put(self, model_id: str, token_ids: tuple[int, ...], cache_state: list[Any]) -> None:
        self.put_calls.append((model_id, token_ids, cache_state))


def test_prompt_cache_orchestrator_prepare_miss() -> None:
    cache = _FakePromptCache()
    orchestrator = PromptCacheOrchestrator(cache)

    plan = orchestrator.prepare("m", [1, 2, 3])

    assert isinstance(plan, PromptCachePlan)
    assert plan.prompt_for_generation == [1, 2, 3]
    assert plan.cache_for_generation is None
    assert plan.prefix_length == 0


def test_prompt_cache_orchestrator_prepare_reuse_hit() -> None:
    reused_cache = ["cache"]
    cache = _FakePromptCache(hit=(reused_cache, 2))
    orchestrator = PromptCacheOrchestrator(cache)

    plan = orchestrator.prepare("m", [1, 2, 3, 4])

    assert plan.prompt_for_generation == [3, 4]
    assert plan.cache_for_generation == reused_cache
    assert plan.prefix_length == 2


def test_prompt_cache_orchestrator_commit_prefers_reused_cache() -> None:
    reused_cache = ["cache"]
    cache = _FakePromptCache(hit=(reused_cache, 1))
    orchestrator = PromptCacheOrchestrator(cache)
    plan = orchestrator.prepare("m", [1, 2, 3])

    orchestrator.commit(plan, generated_ids=[9], final_cache_out=[["new-cache"]])

    assert cache.put_calls == [("m", (1, 2, 3, 9), reused_cache)]


def test_prompt_cache_orchestrator_commit_uses_final_cache_when_no_reuse() -> None:
    cache = _FakePromptCache()
    orchestrator = PromptCacheOrchestrator(cache)
    plan = orchestrator.prepare("m", [1, 2])

    orchestrator.commit(plan, generated_ids=[7, 8], final_cache_out=[["final"]])

    assert cache.put_calls == [("m", (1, 2, 7, 8), ["final"])]
