"""Event-driven memory monitoring for prompt cache (§6.3.1, §6.9).

Checks fire on cache insert and request completion — no polling threads
or background timers. Uses RSS from os.getpid() and optional macOS
memory pressure via sysctl.

Policy: trim_cache → reject_only → shutdown (configurable).
"""

from __future__ import annotations

import logging
import resource
from enum import Enum, auto

import mlx.core as mx

from mlxs._types import MemoryCeilingPolicy

logger = logging.getLogger(__name__)


class MemoryAction(Enum):
    """Result of a memory check."""

    NONE = auto()
    TRIM = auto()
    REJECT = auto()


def get_rss_bytes() -> int:
    """Get current process RSS in bytes (cross-platform)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes; on Linux it's in KB
    return usage.ru_maxrss


def get_max_recommended_bytes() -> int | None:
    """Get MLX device max recommended working set size."""
    try:
        info = mx.device_info()
        return info.get("max_recommended_working_set_size")
    except Exception:
        return None


class MemoryMonitor:
    """Event-driven memory monitor for prompt cache trimming.

    Called explicitly on cache insert and request completion.
    No background threads or timers (§6.3.1).
    """

    __slots__ = (
        "_max_rss_bytes",
        "_policy",
        "_target_ratio",
        "_trim_step",
    )

    def __init__(
        self,
        *,
        max_rss_bytes: int | None = None,
        target_rss_ratio: float = 0.9,
        policy: MemoryCeilingPolicy = MemoryCeilingPolicy.TRIM_CACHE,
        trim_step: int = 1,
    ) -> None:
        self._max_rss_bytes = max_rss_bytes or get_max_recommended_bytes()
        self._target_ratio = target_rss_ratio
        self._policy = policy
        self._trim_step = trim_step

    def check(self) -> MemoryAction:
        """Check if memory pressure requires action.

        Returns:
            MemoryAction indicating what should be done.
        """
        if self._max_rss_bytes is None:
            return MemoryAction.NONE

        rss = get_rss_bytes()
        if rss <= self._max_rss_bytes * self._target_ratio:
            return MemoryAction.NONE

        if self._policy == MemoryCeilingPolicy.REJECT_ONLY:
            return MemoryAction.REJECT

        if self._policy == MemoryCeilingPolicy.TRIM_CACHE:
            return MemoryAction.TRIM

        # shutdown policy — caller handles graceful shutdown
        return MemoryAction.REJECT

    def apply_trim(
        self,
        trim_fn: callable,
    ) -> bool:
        """Run trim loop until RSS is below target or cache is empty.

        Args:
            trim_fn: Callable that trims N entries and returns count removed.

        Returns:
            True if RSS is now below target, False if still above after
            exhausting trim.
        """
        if self._max_rss_bytes is None:
            return True

        target = int(self._max_rss_bytes * self._target_ratio)

        while get_rss_bytes() > target:
            removed = trim_fn(self._trim_step)
            if removed == 0:
                # Nothing left to trim — call mx.clear_cache and check once more
                mx.clear_cache()
                if get_rss_bytes() > target:
                    logger.warning(
                        "Memory still above target after full trim + mx.clear_cache(). "
                        "RSS=%d, target=%d",
                        get_rss_bytes(),
                        target,
                    )
                    return False
                return True

        return True

    @property
    def max_rss_bytes(self) -> int | None:
        return self._max_rss_bytes

    @property
    def policy(self) -> MemoryCeilingPolicy:
        return self._policy
