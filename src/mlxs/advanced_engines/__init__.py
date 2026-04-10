"""Layer 3 / Advanced Engines orchestration boundary above Layers 1–2."""

from __future__ import annotations

from mlxs.advanced_engines.admission import AdmissionDecision, EngineAdmissionPolicy
from mlxs.advanced_engines.prompt_cache import PromptCacheOrchestrator, PromptCachePlan

__all__ = [
    "AdmissionDecision",
    "BatchScheduler",
    "EngineAdmissionPolicy",
    "PromptCacheOrchestrator",
    "PromptCachePlan",
    "draft_tokens",
    "speculative_generate",
]


def __getattr__(name: str) -> object:
    if name == "BatchScheduler":
        from mlxs.advanced_engines.batching import BatchScheduler

        return BatchScheduler
    if name in {"draft_tokens", "speculative_generate"}:
        from mlxs.advanced_engines.speculative import draft_tokens, speculative_generate

        return {
            "draft_tokens": draft_tokens,
            "speculative_generate": speculative_generate,
        }[name]
    raise AttributeError(name)
