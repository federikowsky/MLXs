"""Layer 3 prompt-cache orchestration above the cache substrate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlxs.protocols.prompt_cache import PromptCacheProtocol


@dataclass(frozen=True, slots=True)
class PromptCachePlan:
    """One Layer 3 prompt-cache reuse decision for a generation flow."""

    model_id: str
    full_prompt_token_ids: tuple[int, ...]
    prompt_for_generation: list[int]
    cache_for_generation: list[Any] | None
    prefix_length: int


class PromptCacheOrchestrator:
    """Own cross-request prompt-cache reuse decisions without owning Layer 1 cache state."""

    __slots__ = ("_prompt_cache",)

    def __init__(self, prompt_cache: PromptCacheProtocol) -> None:
        self._prompt_cache = prompt_cache

    def prepare(
        self,
        model_id: str,
        prompt_token_ids: list[int] | tuple[int, ...],
    ) -> PromptCachePlan:
        full_prompt = tuple(prompt_token_ids)
        cache_state, prefix_len = self._prompt_cache.get(model_id, full_prompt)
        suffix_len = len(full_prompt) - prefix_len
        if cache_state is not None and prefix_len > 0 and suffix_len > 0:
            prompt_for_generation = list(full_prompt[prefix_len:])
            cache_for_generation = cache_state
        else:
            prompt_for_generation = list(full_prompt)
            cache_for_generation = None
        return PromptCachePlan(
            model_id=model_id,
            full_prompt_token_ids=full_prompt,
            prompt_for_generation=prompt_for_generation,
            cache_for_generation=cache_for_generation,
            prefix_length=prefix_len,
        )

    def commit(
        self,
        plan: PromptCachePlan,
        *,
        generated_ids: list[int],
        final_cache_out: list[Any],
    ) -> None:
        cache_to_put = (
            plan.cache_for_generation
            if plan.cache_for_generation is not None
            else (final_cache_out[0] if final_cache_out else None)
        )
        if cache_to_put is None:
            return
        new_prefix = plan.full_prompt_token_ids + tuple(generated_ids)
        self._prompt_cache.put(plan.model_id, new_prefix, cache_to_put)
