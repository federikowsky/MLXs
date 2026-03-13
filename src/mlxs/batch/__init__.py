"""Batch module — batch scheduler and continuous batching (§6.4, §9).

Public API: ``BatchScheduler`` — manages concurrent generation sequences.
"""

from mlxs.batch.scheduler import BatchScheduler

__all__ = ["BatchScheduler"]
