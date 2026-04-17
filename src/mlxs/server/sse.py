"""SSE streaming helpers for the server (FR8, §6.7).

Converts TokenEvent streams into Server-Sent Events format compatible
with the OpenAI streaming API convention.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

import anyio
from starlette.responses import StreamingResponse

from mlxs._types import TokenEvent


class DrainFriendlyStreamingResponse(StreamingResponse):
    """StreamingResponse variant that does not cancel on shutdown disconnect."""

    def __init__(
        self,
        content,
        *,
        ignore_disconnect: Any = None,
        on_disconnect: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(content, **kwargs)
        self._ignore_disconnect = ignore_disconnect
        self._on_disconnect = on_disconnect
        self._disconnect_notified = False
        self._stream_finished = False

    def _should_ignore_disconnect(self) -> bool:
        if callable(self._ignore_disconnect):
            return bool(self._ignore_disconnect())
        return bool(self._ignore_disconnect)

    async def _notify_disconnect(self) -> None:
        if self._disconnect_notified:
            return
        self._disconnect_notified = True
        callback = self._on_disconnect
        if callback is None:
            return
        result = callback()
        if hasattr(result, "__await__"):
            await result

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        if scope["type"] == "websocket":
            send = self._wrap_websocket_denial_send(send)
            await self.stream_response(send)
            if self.background is not None:
                await self.background()
            return

        async with anyio.create_task_group() as task_group:
            async def stream_body() -> None:
                try:
                    await self.stream_response(send)
                    self._stream_finished = True
                except OSError:
                    if self._stream_finished:
                        return
                    if self._should_ignore_disconnect():
                        return
                    await self._notify_disconnect()
                finally:
                    task_group.cancel_scope.cancel()

            async def watch_disconnect() -> None:
                while True:
                    message = await receive()
                    if message["type"] != "http.disconnect":
                        continue
                    if self._stream_finished:
                        return
                    if self._should_ignore_disconnect():
                        await anyio.sleep(0.05)
                        continue
                    await self._notify_disconnect()
                    task_group.cancel_scope.cancel()
                    return

            task_group.start_soon(stream_body)
            task_group.start_soon(watch_disconnect)

        if self.background is not None:
            await self.background()


def token_events_to_sse(
    events: Iterator[TokenEvent],
    *,
    model_id: str = "mlxs",
    request_id: str | None = None,
) -> Iterator[str]:
    """Convert a TokenEvent stream to SSE-formatted strings.

    Each yielded string is a complete SSE event (data: {...}\\n\\n).
    Final event is data: [DONE].

    Follows OpenAI chat completions streaming format.
    """
    rid = request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for event in events:
        yield build_sse_chunk(
            event,
            model_id=model_id,
            request_id=rid,
            created=created,
        )

    yield "data: [DONE]\n\n"


def build_sse_chunk(
    event: TokenEvent,
    *,
    model_id: str = "mlxs",
    request_id: str | None = None,
    created: int | None = None,
) -> str:
    """Build one SSE data frame for a single TokenEvent."""
    rid = request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    ts = created if created is not None else int(time.time())
    chunk = _build_chunk(event, model_id=model_id, request_id=rid, created=ts)
    return f"data: {json.dumps(chunk)}\n\n"


def build_sse_error_chunk(
    message: str,
    *,
    status_code: int | None = None,
) -> str:
    """Build one SSE data frame carrying an error payload."""
    error: dict[str, Any] = {"message": message}
    if status_code is not None:
        error["status_code"] = status_code
    return f"data: {json.dumps({'error': error})}\n\n"


def _build_chunk(
    event: TokenEvent,
    *,
    model_id: str,
    request_id: str,
    created: int,
) -> dict[str, Any]:
    """Build an OpenAI-compatible streaming chunk from a TokenEvent."""
    delta: dict[str, str] = {}
    if event.text:
        delta["content"] = event.text

    choice: dict[str, Any] = {
        "index": 0,
        "delta": delta,
    }
    if event.finish_reason is not None:
        choice["finish_reason"] = event.finish_reason.name.lower()
        choice["delta"] = {}

    # Include logprobs if present (FR8)
    if event.logprobs is not None:
        choice["logprobs"] = {
            "content": [
                {
                    "token": event.text,
                    "logprob": event.logprobs.token_logprob,
                    "top_logprobs": [
                        {"token": tlp.token, "logprob": tlp.logprob}
                        for tlp in event.logprobs.top_logprobs
                    ],
                }
            ],
        }

    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [choice],
    }


def _build_logprobs_for_response(events: list[TokenEvent]) -> dict[str, Any]:
    """Build logprobs field for non-streaming response if any event has logprobs."""
    logprobs_content = [
        {
            "token": e.text,
            "logprob": e.logprobs.token_logprob,
            "top_logprobs": [
                {"token": tlp.token, "logprob": tlp.logprob} for tlp in e.logprobs.top_logprobs
            ],
        }
        for e in events
        if e.logprobs is not None
    ]
    if logprobs_content:
        return {"logprobs": {"content": logprobs_content}}
    return {}


def build_completion_response(
    events: list[TokenEvent],
    *,
    model_id: str = "mlxs",
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a non-streaming OpenAI-compatible completion response."""
    rid = request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    text = "".join(e.text for e in events)
    finish_reason = events[-1].finish_reason if events else None
    prompt_tokens = events[0].prompt_tokens if events else 0
    completion_tokens = events[-1].generation_tokens if events else 0

    return {
        "id": rid,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason.name.lower() if finish_reason else None,
                **_build_logprobs_for_response(events),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
