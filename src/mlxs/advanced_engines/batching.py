"""Layer 3 batching and continuous batching boundary."""

from __future__ import annotations

__all__ = ["BatchScheduler"]


def __getattr__(name: str) -> object:
    if name == "BatchScheduler":
        from mlxs.batch.scheduler import BatchScheduler

        return BatchScheduler
    raise AttributeError(name)
