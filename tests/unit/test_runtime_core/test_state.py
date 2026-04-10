"""Tests for Layer 1 runtime state ownership."""

from __future__ import annotations

from typing import Any

from mlxs.runtime_core.state import CoreState


class _FakeCache:
    def __init__(self, label: str) -> None:
        self.label = label


class _FakeModel:
    def __init__(self) -> None:
        self.cache = [_FakeCache("fresh")]

    def make_cache(self) -> list[Any]:
        return self.cache


def test_create_uses_model_cache() -> None:
    model = _FakeModel()
    state = CoreState.create(model)
    assert state.cache is model.cache
    assert state.prompt_tokens == 0
    assert state.generation_tokens == 0


def test_adopt_preserves_existing_cache_identity() -> None:
    cache = [_FakeCache("adopted")]
    state = CoreState.adopt(cache, prompt_tokens=4, generation_tokens=2)
    assert state.cache is cache
    assert state.prompt_tokens == 4
    assert state.generation_tokens == 2


def test_explicit_export_returns_owned_cache() -> None:
    cache = [_FakeCache("owned")]
    state = CoreState.adopt(cache)
    assert state.export_cache() is cache


def test_state_tracks_prompt_and_generation() -> None:
    state = CoreState.adopt([_FakeCache("owned")])
    state.record_prompt(5)
    assert state.prompt_tokens == 5
    assert state.increment_generation() == 1
    assert state.increment_generation() == 2
