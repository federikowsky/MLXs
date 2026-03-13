"""Request queue with backpressure (§6.7).

Bounded async queue for incoming inference requests. Rejects when full
(503) to prevent unbounded memory growth.
"""

from __future__ import annotations

import asyncio
import logging

from mlxs._errors import CapacityExceededError

logger = logging.getLogger(__name__)


class RequestQueue:
    """Bounded async request queue with backpressure.

    When the queue is full, new requests are rejected with
    CapacityExceededError (maps to 503).
    """

    __slots__ = ("_max_size", "_queue", "_timeout")

    def __init__(
        self,
        *,
        max_size: int = 64,
        timeout: float = 300.0,
    ) -> None:
        self._max_size = max_size
        self._timeout = timeout
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_size if max_size > 0 else 0)

    async def put(self, request: dict) -> None:
        """Enqueue a request. Raises CapacityExceededError if full."""
        if self._max_size > 0 and self._queue.qsize() >= self._max_size:
            raise CapacityExceededError(
                f"Request queue full ({self._max_size} pending). Try again later."
            )
        await self._queue.put(request)

    async def get(self) -> dict:
        """Dequeue a request. Blocks until available."""
        return await asyncio.wait_for(self._queue.get(), timeout=self._timeout)

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        return self._max_size > 0 and self._queue.qsize() >= self._max_size
