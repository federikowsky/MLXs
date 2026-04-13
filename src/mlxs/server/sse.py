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

from mlxs._types import TokenEvent


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
