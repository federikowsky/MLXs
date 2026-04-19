"""Layer 4 text-serving host backed by the Layer 3 batch scheduler."""

from __future__ import annotations

import asyncio
import copy
import inspect
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mlxs.advanced_engines.prompt_cache import PromptCacheOrchestrator, PromptCachePlan
from mlxs._types import GenerateOptions, TokenEvent
from mlxs.batch.scheduler import BatchScheduler
from mlxs.protocols.batch import BatchSchedulerProtocol


@dataclass(slots=True)
class _PendingResult:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[list[TokenEvent]]
    cache_plan: PromptCachePlan | None
    events: list[TokenEvent]
    stream_queue: asyncio.Queue[TokenEvent | None] | None


def _supports_kwarg(callable_obj: Callable[..., Any], name: str) -> bool:
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


class BatchServingHost:
    """Layer 4-owned text generation host using a single batch scheduler worker.

    The host preserves current product semantics by returning the same
    ``list[TokenEvent]`` surface the HTTP path already expects. It does not own
    transport concerns and falls back to the legacy single-request path where
    the caller chooses not to use it.
    """

    __slots__ = (
        "_commands",
        "_lock",
        "_model",
        "_model_id",
        "_pending",
        "_prompt_cache_orchestrator",
        "_scheduler",
        "_scheduler_add_supports_cache_state",
        "_scheduler_add_supports_prompt_token_count",
        "_scheduler_drains_cache_state",
        "_stop",
        "_thread",
        "_tokenizer",
    )

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prefill_batch_size: int,
        completion_batch_size: int,
        prefill_step_size: int,
        scheduler: BatchSchedulerProtocol | None = None,
        prompt_cache_orchestrator: PromptCacheOrchestrator | None = None,
        model_id: str | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._prompt_cache_orchestrator = prompt_cache_orchestrator
        self._model_id = model_id
        if scheduler is None:
            scheduler_kwargs = {
                "completion_batch_size": completion_batch_size,
                "prefill_step_size": prefill_step_size,
            }
            if _supports_kwarg(BatchScheduler, "prefill_batch_size"):
                scheduler_kwargs["prefill_batch_size"] = prefill_batch_size
            self._scheduler = BatchScheduler(**scheduler_kwargs)
        else:
            self._scheduler = scheduler
        self._scheduler_add_supports_cache_state = _supports_kwarg(
            self._scheduler.add,
            "cache_state",
        )
        self._scheduler_add_supports_prompt_token_count = _supports_kwarg(
            self._scheduler.add,
            "prompt_token_count",
        )
        self._scheduler_drains_cache_state = (
            self._scheduler_add_supports_cache_state
            and self._scheduler_add_supports_prompt_token_count
        )
        self._commands: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._pending: dict[str, _PendingResult] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return True

    async def execute(
        self,
        prompt: str | list[int],
        options: GenerateOptions,
    ) -> list[TokenEvent]:
        """Execute one text-only request through the scheduler host."""
        request_id, future, _stream_queue = self._start_request(prompt, options, stream=False)
        try:
            return await future
        except asyncio.CancelledError:
            self._commands.put(("remove", request_id))
            raise

    async def stream_execute(
        self,
        prompt: str | list[int],
        options: GenerateOptions,
    ):
        """Execute one request and yield TokenEvents incrementally."""
        request_id, future, stream_queue = self._start_request(prompt, options, stream=True)
        assert stream_queue is not None
        try:
            while True:
                event = await stream_queue.get()
                if event is None:
                    break
                yield event
            await future
        except asyncio.CancelledError:
            self._commands.put(("remove", request_id))
            raise

    def shutdown(self) -> None:
        """Stop the worker thread if it has been started."""
        self._stop.set()
        self._commands.put(("stop",))
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            self._drain_commands(block=not self._has_work())
            if not self._has_work():
                continue

            step_events = self._scheduler.step()
            for request_id, events in step_events.items():
                self._publish_events(request_id, events)
            for drained in self._scheduler.drain():
                if len(drained) == 3:
                    request_id, events, final_cache = drained
                else:
                    request_id, events = drained
                    final_cache = None
                self._complete_request(request_id, events, final_cache)

    def _drain_commands(self, *, block: bool) -> None:
        if block:
            try:
                command = self._commands.get(timeout=0.05)
            except queue.Empty:
                return
            self._handle_command(command)

        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            self._handle_command(command)

    def _handle_command(self, command: tuple[Any, ...]) -> None:
        match command:
            case ("add", request_id, prompt, options, add_kwargs):
                self._scheduler.add(
                    request_id,
                    self._model,
                    self._tokenizer,
                    prompt,
                    options,
                    **add_kwargs,
                )
            case ("remove", request_id):
                self._scheduler.remove(request_id)
            case ("stop",):
                return
            case _:
                raise ValueError(f"Unknown batch host command: {command!r}")

    def _complete_request(
        self,
        request_id: str,
        events: list[TokenEvent],
        final_cache: list[Any] | None,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        if not pending.events and events:
            pending.events.extend(events)
        self._commit_prompt_cache(pending.cache_plan, pending.events, final_cache)

        def _resolve() -> None:
            if pending.future.cancelled() or pending.future.done():
                return
            pending.future.set_result(list(pending.events))

        pending.loop.call_soon_threadsafe(_resolve)
        if pending.stream_queue is not None:
            pending.loop.call_soon_threadsafe(pending.stream_queue.put_nowait, None)

    def _has_work(self) -> bool:
        return self._scheduler.pending_count > 0 or self._scheduler.active_count > 0

    def _prepare_prompt_cache_plan(
        self,
        prompt: str | list[int],
    ) -> PromptCachePlan | None:
        if (
            self._prompt_cache_orchestrator is None
            or self._model_id is None
            or not self._scheduler_add_supports_cache_state
            or not self._scheduler_add_supports_prompt_token_count
            or not self._scheduler_drains_cache_state
        ):
            return None
        token_ids = (
            self._tokenizer.encode(prompt)
            if isinstance(prompt, str)
            else list(prompt)
        )
        return self._prompt_cache_orchestrator.prepare(self._model_id, token_ids)

    def _commit_prompt_cache(
        self,
        cache_plan: PromptCachePlan | None,
        events: list[TokenEvent],
        final_cache: list[Any] | None,
    ) -> None:
        if cache_plan is None or self._prompt_cache_orchestrator is None:
            return
        cache_for_commit: list[Any] | None = None
        if final_cache is not None:
            cache_for_commit = copy.deepcopy(final_cache)
            generated_count = len(events)
            if generated_count > 0:
                for layer in cache_for_commit:
                    trim = getattr(layer, "trim", None)
                    if callable(trim):
                        trim(generated_count)
        self._prompt_cache_orchestrator.commit(
            cache_plan,
            generated_ids=[],
            final_cache_out=[cache_for_commit] if cache_for_commit is not None else [],
        )

    def _publish_events(self, request_id: str, events: list[TokenEvent]) -> None:
        with self._lock:
            pending = self._pending.get(request_id)
        if pending is None:
            return
        pending.events.extend(events)
        if pending.stream_queue is None:
            return
        for event in events:
            pending.loop.call_soon_threadsafe(pending.stream_queue.put_nowait, event)

    def _start_request(
        self,
        prompt: str | list[int],
        options: GenerateOptions,
        *,
        stream: bool,
    ) -> tuple[str, asyncio.Future[list[TokenEvent]], asyncio.Queue[TokenEvent | None] | None]:
        request_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[TokenEvent]] = loop.create_future()
        stream_queue: asyncio.Queue[TokenEvent | None] | None = (
            asyncio.Queue() if stream else None
        )
        cache_plan = self._prepare_prompt_cache_plan(prompt)
        with self._lock:
            self._pending[request_id] = _PendingResult(
                loop=loop,
                future=future,
                cache_plan=cache_plan,
                events=[],
                stream_queue=stream_queue,
            )
        self._ensure_thread()
        if cache_plan is None:
            scheduler_prompt = prompt
            add_kwargs: dict[str, Any] = {}
        else:
            scheduler_prompt = cache_plan.prompt_for_generation
            add_kwargs = {
                "cache_state": cache_plan.cache_for_generation,
                "prompt_token_count": len(cache_plan.full_prompt_token_ids),
            }
        self._commands.put(("add", request_id, scheduler_prompt, options, add_kwargs))
        return request_id, future, stream_queue


__all__ = ["BatchServingHost"]
