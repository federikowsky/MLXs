"""Request queue with backpressure (§6.7).

Bounded async queue for incoming inference requests. Rejects when full
(503) to prevent unbounded memory growth.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque

from mlxs._errors import CapacityExceededError, RequestTimeoutError

logger = logging.getLogger(__name__)


class RequestQueue:
    """Bounded async request queue with backpressure.

    When the queue is full, new requests are rejected with
    CapacityExceededError (maps to 503).
    """

    __slots__ = ("_active", "_max_concurrent", "_max_size", "_pending", "_timeout")

    def __init__(
        self,
        *,
        max_size: int = 64,
        max_concurrent: int = 16,
        timeout: float = 300.0,
    ) -> None:
        self._active = 0
        self._max_concurrent = max_concurrent
        self._max_size = max_size
        self._timeout = timeout
        self._pending: deque[tuple[dict, asyncio.Future[dict]]] = deque()

    async def put(self, request: dict, *, timeout: float | None = None) -> None:
        """Admit a request or queue it pending a free execution slot."""
        if self._active < self._max_concurrent:
            self._active += 1
            return

        if self._max_size > 0 and len(self._pending) >= self._max_size:
            raise CapacityExceededError(
                f"Request queue full ({self._max_size} pending). Try again later."
            )

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[dict] = loop.create_future()
        entry = (request, waiter)
        self._pending.append(entry)
        wait_timeout = self._timeout if timeout is None else timeout
        try:
            if wait_timeout is None:
                await waiter
            else:
                await asyncio.wait_for(waiter, timeout=wait_timeout)
        except asyncio.TimeoutError as exc:
            try:
                self._pending.remove(entry)
            except ValueError:
                pass
            raise RequestTimeoutError(f"Request timed out after {wait_timeout} seconds.") from exc

    async def get(self) -> dict:
        """Release one active slot and admit the next pending request if any."""
        if self._pending:
            request, waiter = self._pending.popleft()
            if not waiter.done():
                waiter.set_result(request)
            return request
        if self._active > 0:
            self._active -= 1
        return {}

    @property
    def size(self) -> int:
        return len(self._pending)

    @property
    def is_full(self) -> bool:
        return self._max_size > 0 and len(self._pending) >= self._max_size

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def pending_count(self) -> int:
        return len(self._pending)
