"""Layer 3 speculative decoding orchestration boundary."""

from __future__ import annotations

__all__ = ["draft_tokens", "speculative_generate"]


def __getattr__(name: str) -> object:
    if name in {"draft_tokens", "speculative_generate"}:
        from mlxs.speculative.draft import draft_tokens
        from mlxs.speculative.verify import speculative_generate

        return {
            "draft_tokens": draft_tokens,
            "speculative_generate": speculative_generate,
        }[name]
    raise AttributeError(name)
