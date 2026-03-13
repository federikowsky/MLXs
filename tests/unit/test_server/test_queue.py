"""Tests for request queue with backpressure (§6.7).

Patterns: happy path, boundary, negative path, stateful lifecycle,
recovery/resilience.
"""

from __future__ import annotations

import asyncio

import pytest

from mlxs._errors import CapacityExceededError
from mlxs.server.queue import RequestQueue


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestQueueHappyPath:
    def test_put_and_get(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=10)
            await q.put({"id": 1})
            result = await q.get()
            assert result == {"id": 1}

        asyncio.run(_test())

    def test_fifo_order(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=10)
            for i in range(5):
                await q.put({"id": i})
            for i in range(5):
                result = await q.get()
                assert result["id"] == i

        asyncio.run(_test())

    def test_size_property(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=10)
            assert q.size == 0
            await q.put({"id": 1})
            assert q.size == 1
            await q.get()
            assert q.size == 0

        asyncio.run(_test())


class TestQueueBoundary:
    def test_max_size_one(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=1)
            await q.put({"id": 1})
            assert q.is_full
            with pytest.raises(CapacityExceededError):
                await q.put({"id": 2})

        asyncio.run(_test())

    def test_exactly_at_capacity(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=3)
            for i in range(3):
                await q.put({"id": i})
            assert q.is_full
            assert q.size == 3

        asyncio.run(_test())

    def test_unbounded_queue(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=0)  # 0 = unbounded
            for i in range(100):
                await q.put({"id": i})
            assert q.size == 100
            assert not q.is_full

        asyncio.run(_test())


class TestQueueNegativePath:
    def test_reject_when_full(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=2)
            await q.put({"id": 1})
            await q.put({"id": 2})
            with pytest.raises(CapacityExceededError, match="queue full"):
                await q.put({"id": 3})

        asyncio.run(_test())

    def test_error_is_mlxs_error(self) -> None:
        """CapacityExceededError has correct status_hint for HTTP mapping."""
        err = CapacityExceededError("full")
        assert err.status_hint == 503


class TestQueueStateful:
    def test_capacity_restored_after_get(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=1)
            await q.put({"id": 1})
            assert q.is_full
            await q.get()
            assert not q.is_full
            await q.put({"id": 2})  # should not raise
            assert q.size == 1

        asyncio.run(_test())

    def test_multiple_fill_drain_cycles(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=2)
            for cycle in range(3):
                await q.put({"cycle": cycle, "idx": 0})
                await q.put({"cycle": cycle, "idx": 1})
                assert q.is_full
                await q.get()
                await q.get()
                assert q.size == 0

        asyncio.run(_test())
