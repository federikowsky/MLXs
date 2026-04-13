"""Tests for Layer 4 request admission semantics (§6.7, Phase 5)."""

from __future__ import annotations

import asyncio

import pytest

from mlxs._errors import CapacityExceededError, RequestTimeoutError
from mlxs.server.queue import RequestQueue


class TestQueueHappyPath:
    def test_admits_immediately_when_active_slot_available(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=10, max_concurrent=2)
            await q.put({"id": 1})
            assert q.active_count == 1
            assert q.pending_count == 0
            assert q.size == 0
            await q.get()
            assert q.active_count == 0

        asyncio.run(_test())

    def test_pending_request_is_admitted_on_release(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=10, max_concurrent=1)
            await q.put({"id": 1})
            waiting = asyncio.create_task(q.put({"id": 2}, timeout=0.1))
            await asyncio.sleep(0)
            assert q.active_count == 1
            assert q.pending_count == 1
            released = await q.get()
            await waiting
            assert released == {"id": 2}
            assert q.active_count == 1
            assert q.pending_count == 0
            await q.get()
            assert q.active_count == 0

        asyncio.run(_test())

    def test_pending_admission_is_fifo(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=10, max_concurrent=1)
            await q.put({"id": 1})
            order: list[int] = []

            async def _wait(request_id: int) -> None:
                await q.put({"id": request_id}, timeout=0.2)
                order.append(request_id)

            second = asyncio.create_task(_wait(2))
            third = asyncio.create_task(_wait(3))
            await asyncio.sleep(0)
            assert q.pending_count == 2

            admitted = await q.get()
            assert admitted == {"id": 2}
            await second
            assert order == [2]

            admitted = await q.get()
            assert admitted == {"id": 3}
            await third
            assert order == [2, 3]

            await q.get()
            assert q.active_count == 0

        asyncio.run(_test())


class TestQueueBoundary:
    def test_rejects_when_pending_queue_is_full(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=1, max_concurrent=1)
            await q.put({"id": 1})
            waiting = asyncio.create_task(q.put({"id": 2}, timeout=0.1))
            await asyncio.sleep(0)
            assert q.is_full
            with pytest.raises(CapacityExceededError, match="queue full"):
                await q.put({"id": 3}, timeout=0.1)
            admitted = await q.get()
            assert admitted == {"id": 2}
            await waiting
            await q.get()

        asyncio.run(_test())

    def test_tracks_peak_active_and_pending_counts(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=2, max_concurrent=1)
            await q.put({"id": 1})
            waiting_2 = asyncio.create_task(q.put({"id": 2}, timeout=0.2))
            waiting_3 = asyncio.create_task(q.put({"id": 3}, timeout=0.2))
            await asyncio.sleep(0)
            assert q.peak_active_count == 1
            assert q.peak_pending_count == 2
            admitted = await q.get()
            assert admitted == {"id": 2}
            await waiting_2
            admitted = await q.get()
            assert admitted == {"id": 3}
            await waiting_3
            await q.get()
            assert q.active_count == 0
            assert q.pending_count == 0

        asyncio.run(_test())

    def test_unbounded_pending_queue_respects_active_cap(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=0, max_concurrent=1)
            await q.put({"id": 1})
            tasks = [asyncio.create_task(q.put({"id": i}, timeout=0.2)) for i in range(2, 12)]
            await asyncio.sleep(0)
            assert q.active_count == 1
            assert q.pending_count == 10
            assert not q.is_full
            for _ in range(10):
                await q.get()
            await asyncio.gather(*tasks)
            await q.get()
            assert q.active_count == 0
            assert q.pending_count == 0

        asyncio.run(_test())


class TestQueueTimeouts:
    def test_pending_request_times_out_while_waiting_for_slot(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=1, max_concurrent=1, timeout=0.01)
            await q.put({"id": 1})
            with pytest.raises(RequestTimeoutError, match="timed out"):
                await q.put({"id": 2})
            assert q.active_count == 1
            assert q.pending_count == 0
            await q.get()
            assert q.active_count == 0

        asyncio.run(_test())

    def test_capacity_restores_after_timed_out_waiter(self) -> None:
        async def _test() -> None:
            q = RequestQueue(max_size=1, max_concurrent=1, timeout=0.01)
            await q.put({"id": 1})
            with pytest.raises(RequestTimeoutError):
                await q.put({"id": 2})
            waiting = asyncio.create_task(q.put({"id": 3}, timeout=0.1))
            await asyncio.sleep(0)
            admitted = await q.get()
            assert admitted == {"id": 3}
            await waiting
            await q.get()
            assert q.active_count == 0
            assert q.pending_count == 0

        asyncio.run(_test())


class TestQueueErrors:
    def test_capacity_exceeded_error_is_http_compatible(self) -> None:
        err = CapacityExceededError("full")
        assert err.status_hint == 503

    def test_request_timeout_error_is_http_compatible(self) -> None:
        err = RequestTimeoutError("timeout")
        assert err.status_hint == 503
