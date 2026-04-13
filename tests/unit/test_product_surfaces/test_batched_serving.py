from __future__ import annotations

import asyncio
import time

import pytest

from mlxs.advanced_engines.prompt_cache import PromptCachePlan
from mlxs._types import FinishReason, GenerateOptions, TokenEvent
from mlxs.product_surfaces.batched_serving import BatchServingHost


class _FakeScheduler:
    def __init__(self) -> None:
        self.pending_count = 0
        self.active_count = 0
        self.added: list[str] = []
        self.removed: list[str] = []
        self.add_calls: list[dict[str, object]] = []
        self._finished: dict[str, tuple[list[TokenEvent], list[object] | None]] = {}

    def add(  # type: ignore[no-untyped-def]
        self,
        request_id,
        model,
        tokenizer,
        prompt,
        options,
        *,
        cache_state=None,
        prompt_token_count=None,
    ) -> None:
        del model, tokenizer, options
        self.added.append(str(prompt))
        self.add_calls.append(
            {
                "prompt": prompt,
                "cache_state": cache_state,
                "prompt_token_count": prompt_token_count,
            }
        )
        self.pending_count = 1
        self._finished[request_id] = (
            [TokenEvent(token_id=1, text="ok", finish_reason=FinishReason.STOP)],
            None,
        )

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
        for request_id, (events, final_cache) in finished:
            yield request_id, events, final_cache


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
        def add(self, request_id, model, tokenizer, prompt, options, *, cache_state=None, prompt_token_count=None) -> None:  # type: ignore[no-untyped-def]
            del model, tokenizer, prompt, options, cache_state, prompt_token_count
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


def test_batch_serving_host_uses_prompt_cache_plan_and_commits_prompt_only_cache() -> None:
    class _PromptCacheScheduler(_FakeScheduler):
        def add(  # type: ignore[no-untyped-def]
            self,
            request_id,
            model,
            tokenizer,
            prompt,
            options,
            *,
            cache_state=None,
            prompt_token_count=None,
        ) -> None:
            super().add(
                request_id,
                model,
                tokenizer,
                prompt,
                options,
                cache_state=cache_state,
                prompt_token_count=prompt_token_count,
            )
            self._finished[request_id] = (
                [
                    TokenEvent(token_id=7, text="x"),
                    TokenEvent(token_id=8, text="y", finish_reason=FinishReason.STOP),
                ],
                [_TrimCache(6)],
            )

    class _FakePromptCacheOrchestrator:
        def __init__(self) -> None:
            self.prepare_calls: list[tuple[str, list[int]]] = []
            self.commit_calls: list[tuple[PromptCachePlan, list[int], list[object]]] = []

        def prepare(self, model_id: str, prompt_token_ids: list[int]) -> PromptCachePlan:
            self.prepare_calls.append((model_id, prompt_token_ids))
            return PromptCachePlan(
                model_id=model_id,
                full_prompt_token_ids=(1, 2, 3, 4),
                prompt_for_generation=[3, 4],
                cache_for_generation=[_TrimCache(7)],
                prefix_length=2,
            )

        def commit(self, plan: PromptCachePlan, *, generated_ids: list[int], final_cache_out: list[object]) -> None:
            self.commit_calls.append((plan, generated_ids, final_cache_out))

    class _TrimCache:
        def __init__(self, offset: int) -> None:
            self.offset = offset

        def trim(self, n: int) -> int:
            self.offset -= n
            return n

    scheduler = _PromptCacheScheduler()
    orchestrator = _FakePromptCacheOrchestrator()
    host = BatchServingHost(
        model=object(),
        tokenizer=type("Tok", (), {"encode": staticmethod(lambda prompt: [1, 2, 3, 4])})(),
        prefill_batch_size=1,
        completion_batch_size=4,
        prefill_step_size=2048,
        scheduler=scheduler,
        prompt_cache_orchestrator=orchestrator,
        model_id="model-x",
    )

    events = asyncio.run(host.execute("prompt", GenerateOptions()))

    assert [event.text for event in events] == ["x", "y"]
    assert orchestrator.prepare_calls == [("model-x", [1, 2, 3, 4])]
    assert scheduler.add_calls[0]["prompt"] == [3, 4]
    assert scheduler.add_calls[0]["prompt_token_count"] == 4
    assert len(orchestrator.commit_calls) == 1
    _plan, generated_ids, final_cache_out = orchestrator.commit_calls[0]
    assert generated_ids == []
    committed_cache = final_cache_out[0][0]
    assert committed_cache.offset == 4
    host.shutdown()
