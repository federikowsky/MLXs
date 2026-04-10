"""Layer 1 runtime state ownership."""

from __future__ import annotations

from dataclasses import dataclass

from mlxs.protocols.cache import CacheProtocol
from mlxs.protocols.model import ModelProtocol


@dataclass(slots=True)
class CoreState:
    """Sole owner of active mutable runtime state during Layer 1 execution."""

    cache: list[CacheProtocol]
    prompt_tokens: int = 0
    generation_tokens: int = 0

    @classmethod
    def create(cls, model: ModelProtocol) -> CoreState:
        """Create a fresh active cache owned by Layer 1."""
        return cls(cache=model.make_cache())

    @classmethod
    def adopt(
        cls,
        cache: list[CacheProtocol],
        *,
        prompt_tokens: int = 0,
        generation_tokens: int = 0,
    ) -> CoreState:
        """Adopt an existing active cache as Layer 1-owned state."""
        return cls(
            cache=cache,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
        )

    def record_prompt(self, prompt_tokens: int) -> None:
        """Store the current prompt length after prefill ownership begins."""
        self.prompt_tokens = prompt_tokens

    def increment_generation(self) -> int:
        """Advance the generated-token counter and return the new value."""
        self.generation_tokens += 1
        return self.generation_tokens

    def export_cache(self) -> list[CacheProtocol]:
        """Explicitly export the owned cache handle."""
        return self.cache
