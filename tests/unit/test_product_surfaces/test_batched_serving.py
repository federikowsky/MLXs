from __future__ import annotations

import asyncio
import time

import pytest

from mlxs._types import FinishReason, GenerateOptions, TokenEvent
from mlxs.product_surfaces.batched_serving import BatchServingHost


class _FakeScheduler:
    def __init__(self) -> None:
        self.pending_count = 0
        self.active_count = 0
        self.added: list[str] = []
        self.removed: list[str] = []
        self._finished: dict[str, list[TokenEvent]] = {}

    def add(self, request_id, model, tokenizer, prompt, options) -> None:  # type: ignore[no-untyped-def]
        del model, tokenizer, options
        self.added.append(str(prompt))
        self.pending_count = 1
        self._finished[request_id] = [
            TokenEvent(token_id=1, text="ok", finish_reason=FinishReason.STOP)
        ]

    def remove(self, request_id: str) -> None:
        self.removed.append(request_id)
        self.pending_count = 0
        self.active_count = 0

    def step(self) -> dict[str, list[TokenEvent]]:
        self.pending_count = 0
        self.active_count = 1 if self._finished else 0
        return {}

    def drain(self):
        finished = list(self._finished.items())
        self._finished.clear()
        self.active_count = 0
        yield from finished


def test_batch_serving_host_executes_and_returns_events() -> None:
    host = BatchServingHost(
        model=object(),
        tokenizer=object(),
        prefill_batch_size=1,
        completion_batch_size=4,
        prefill_step_size=2048,
        scheduler=_FakeScheduler(),
    )

    events = asyncio.run(host.execute("prompt", GenerateOptions()))

    assert [event.text for event in events] == ["ok"]
    host.shutdown()


def test_batch_serving_host_cancels_pending_request() -> None:
    class _NeverFinishScheduler(_FakeScheduler):
        def add(self, request_id, model, tokenizer, prompt, options) -> None:  # type: ignore[no-untyped-def]
            del model, tokenizer, prompt, options
            self.pending_count = 1
            self.request_id = request_id

        def step(self) -> dict[str, list[TokenEvent]]:
            self.pending_count = 0
            self.active_count = 1
            return {}

        def drain(self):
            return iter(())

    scheduler = _NeverFinishScheduler()
    host = BatchServingHost(
        model=object(),
        tokenizer=object(),
        prefill_batch_size=1,
        completion_batch_size=4,
        prefill_step_size=2048,
        scheduler=scheduler,
    )

    async def _run() -> None:
        task = asyncio.create_task(host.execute("prompt", GenerateOptions()))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    time.sleep(0.05)

    assert scheduler.removed
    host.shutdown()
