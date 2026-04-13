"""Layer 4 text-serving host backed by the Layer 3 batch scheduler."""

from __future__ import annotations

import asyncio
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mlxs._types import GenerateOptions, TokenEvent
from mlxs.batch.scheduler import BatchScheduler
from mlxs.protocols.batch import BatchSchedulerProtocol


@dataclass(slots=True)
class _PendingResult:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[list[TokenEvent]]


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
        "_pending",
        "_scheduler",
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
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._scheduler = scheduler or BatchScheduler(
            prefill_batch_size=prefill_batch_size,
            completion_batch_size=completion_batch_size,
            prefill_step_size=prefill_step_size,
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
        request_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[TokenEvent]] = loop.create_future()
        with self._lock:
            self._pending[request_id] = _PendingResult(loop=loop, future=future)
        self._ensure_thread()
        self._commands.put(("add", request_id, prompt, options))
        try:
            return await future
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

            self._scheduler.step()
            for request_id, events in self._scheduler.drain():
                self._complete_request(request_id, events)

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
            case ("add", request_id, prompt, options):
                self._scheduler.add(
                    request_id,
                    self._model,
                    self._tokenizer,
                    prompt,
                    options,
                )
            case ("remove", request_id):
                self._scheduler.remove(request_id)
            case ("stop",):
                return
            case _:
                raise ValueError(f"Unknown batch host command: {command!r}")

    def _complete_request(self, request_id: str, events: list[TokenEvent]) -> None:
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return

        def _resolve() -> None:
            if pending.future.cancelled() or pending.future.done():
                return
            pending.future.set_result(events)

        pending.loop.call_soon_threadsafe(_resolve)

    def _has_work(self) -> bool:
        return self._scheduler.pending_count > 0 or self._scheduler.active_count > 0


__all__ = ["BatchServingHost"]
