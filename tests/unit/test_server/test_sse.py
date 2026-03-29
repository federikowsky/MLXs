"""Tests for SSE streaming helpers (FR8, §6.7).

Patterns: happy path, edge cases, boundary, alternate flows,
corner cases, negative path.
"""

from __future__ import annotations

import json
import time

import pytest

from mlxs._types import FinishReason, TokenEvent, TokenLogprobs, TopLogprob
from mlxs.server.sse import (
    build_completion_response,
    format_token_event_sse_line,
    token_event_to_openai_stream_json,
    token_events_to_sse,
)

# =============================================================================
# token_events_to_sse — streaming
# =============================================================================


class TestFormatTokenEventSseLine:
    def test_payload_matches_token_events_to_sse_first_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(time, "time", lambda: 100.0)
        ev = TokenEvent(token_id=1, text="Hello")
        rid = "chatcmpl-fixed"
        first = next(token_events_to_sse(iter([ev]), model_id="test", request_id=rid))
        line = format_token_event_sse_line(
            ev, model_id="test", request_id=rid, created=100
        )
        assert first == line

    def test_opensai_chunk_object_type(self) -> None:
        ev = TokenEvent(token_id=1, text="x", finish_reason=FinishReason.STOP)
        line = format_token_event_sse_line(
            ev, model_id="m", request_id="r", created=1
        )
        data = json.loads(line.removeprefix("data: ").strip())
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_stream_json_matches_embedded_json_in_sse_line(self) -> None:
        ev = TokenEvent(token_id=1, text="z", finish_reason=FinishReason.STOP)
        inner = token_event_to_openai_stream_json(
            ev, model_id="m", request_id="r", created=9
        )
        outer = format_token_event_sse_line(
            ev, model_id="m", request_id="r", created=9
        )
        assert outer == f"data: {inner}\n\n"


class TestSSEStreamHappyPath:
    def test_basic_stream(self) -> None:
        events = [
            TokenEvent(token_id=1, text="Hello"),
            TokenEvent(token_id=2, text=" world", finish_reason=FinishReason.STOP),
        ]
        chunks = list(token_events_to_sse(iter(events), model_id="test"))
        assert len(chunks) == 3  # 2 events + [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"

        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["model"] == "test"

    def test_stream_with_logprobs(self) -> None:
        lp = TokenLogprobs(
            token_logprob=-0.5,
            top_logprobs=(
                TopLogprob(token_id=1, token="Hello", logprob=-0.5),
                TopLogprob(token_id=2, token="Hi", logprob=-1.2),
            ),
        )
        events = [TokenEvent(token_id=1, text="Hello", logprobs=lp)]
        chunks = list(token_events_to_sse(iter(events)))
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        choice = data["choices"][0]
        assert "logprobs" in choice
        assert choice["logprobs"]["content"][0]["logprob"] == -0.5
        assert len(choice["logprobs"]["content"][0]["top_logprobs"]) == 2

    def test_all_chunks_are_valid_sse(self) -> None:
        events = [TokenEvent(token_id=i, text=f"t{i}") for i in range(5)]
        events[-1].finish_reason = FinishReason.STOP
        chunks = list(token_events_to_sse(iter(events)))
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

    def test_request_id_consistent(self) -> None:
        events = [
            TokenEvent(token_id=1, text="a"),
            TokenEvent(token_id=2, text="b", finish_reason=FinishReason.STOP),
        ]
        chunks = list(token_events_to_sse(iter(events)))
        ids = [
            json.loads(c.removeprefix("data: ").strip())["id"]
            for c in chunks
            if c != "data: [DONE]\n\n"
        ]
        assert all(i == ids[0] for i in ids)

    def test_custom_request_id(self) -> None:
        events = [TokenEvent(token_id=1, text="x", finish_reason=FinishReason.STOP)]
        chunks = list(token_events_to_sse(iter(events), request_id="custom-123"))
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["id"] == "custom-123"


class TestSSEStreamAlternateFlows:
    def test_finish_reason_clears_delta(self) -> None:
        events = [
            TokenEvent(token_id=1, text="done", finish_reason=FinishReason.LENGTH),
        ]
        chunks = list(token_events_to_sse(iter(events)))
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["choices"][0]["finish_reason"] == "length"
        assert data["choices"][0]["delta"] == {}

    def test_finish_reason_stop(self) -> None:
        events = [TokenEvent(token_id=1, text="x", finish_reason=FinishReason.STOP)]
        chunks = list(token_events_to_sse(iter(events)))
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_finish_reason_tool_calls(self) -> None:
        events = [TokenEvent(token_id=1, text="", finish_reason=FinishReason.TOOL_CALLS)]
        chunks = list(token_events_to_sse(iter(events)))
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_no_logprobs_means_no_logprobs_field(self) -> None:
        events = [TokenEvent(token_id=1, text="x")]
        chunks = list(token_events_to_sse(iter(events)))
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert "logprobs" not in data["choices"][0]


class TestSSEStreamEdgeCases:
    def test_empty_text_token(self) -> None:
        events = [TokenEvent(token_id=1, text="")]
        chunks = list(token_events_to_sse(iter(events)))
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert "content" not in data["choices"][0]["delta"]

    def test_logprobs_with_empty_top_logprobs(self) -> None:
        lp = TokenLogprobs(token_logprob=-1.0, top_logprobs=())
        events = [TokenEvent(token_id=1, text="x", logprobs=lp)]
        chunks = list(token_events_to_sse(iter(events)))
        data = json.loads(chunks[0].removeprefix("data: ").strip())
        assert data["choices"][0]["logprobs"]["content"][0]["top_logprobs"] == []


class TestSSEStreamBoundary:
    def test_single_event_stream(self) -> None:
        events = [TokenEvent(token_id=0, text="only", finish_reason=FinishReason.STOP)]
        chunks = list(token_events_to_sse(iter(events)))
        assert len(chunks) == 2  # 1 event + [DONE]

    def test_empty_event_stream(self) -> None:
        chunks = list(token_events_to_sse(iter([])))
        assert len(chunks) == 1
        assert chunks[0] == "data: [DONE]\n\n"


# =============================================================================
# build_completion_response — non-streaming
# =============================================================================


class TestCompletionResponseHappyPath:
    def test_basic_response(self) -> None:
        events = [
            TokenEvent(token_id=1, text="Hello", prompt_tokens=5, generation_tokens=1),
            TokenEvent(
                token_id=2,
                text=" world",
                finish_reason=FinishReason.STOP,
                prompt_tokens=5,
                generation_tokens=2,
            ),
        ]
        resp = build_completion_response(events, model_id="test")
        assert resp["choices"][0]["message"]["content"] == "Hello world"
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert resp["usage"]["prompt_tokens"] == 5
        assert resp["usage"]["completion_tokens"] == 2
        assert resp["usage"]["total_tokens"] == 7

    def test_response_object_type(self) -> None:
        events = [TokenEvent(token_id=1, text="x", finish_reason=FinishReason.STOP)]
        resp = build_completion_response(events)
        assert resp["object"] == "chat.completion"


class TestCompletionResponseWithLogprobs:
    def test_response_with_logprobs(self) -> None:
        lp = TokenLogprobs(token_logprob=-0.3, top_logprobs=())
        events = [
            TokenEvent(token_id=1, text="Hi", logprobs=lp, finish_reason=FinishReason.STOP),
        ]
        resp = build_completion_response(events)
        assert "logprobs" in resp["choices"][0]
        assert resp["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.3

    def test_response_without_logprobs(self) -> None:
        events = [
            TokenEvent(token_id=1, text="Hi", finish_reason=FinishReason.STOP),
        ]
        resp = build_completion_response(events)
        assert "logprobs" not in resp["choices"][0]

    def test_partial_logprobs(self) -> None:
        """Some events have logprobs, some don't — only those with logprobs appear."""
        lp = TokenLogprobs(token_logprob=-0.5, top_logprobs=())
        events = [
            TokenEvent(token_id=1, text="a"),
            TokenEvent(token_id=2, text="b", logprobs=lp),
            TokenEvent(token_id=3, text="c", finish_reason=FinishReason.STOP),
        ]
        resp = build_completion_response(events)
        assert "logprobs" in resp["choices"][0]
        assert len(resp["choices"][0]["logprobs"]["content"]) == 1

    def test_logprobs_with_top_logprobs(self) -> None:
        lp = TokenLogprobs(
            token_logprob=-0.2,
            top_logprobs=(
                TopLogprob(token_id=10, token="the", logprob=-0.2),
                TopLogprob(token_id=20, token="a", logprob=-1.5),
            ),
        )
        events = [
            TokenEvent(token_id=10, text="the", logprobs=lp, finish_reason=FinishReason.STOP),
        ]
        resp = build_completion_response(events)
        top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        assert len(top) == 2
        assert top[0]["token"] == "the"
        assert top[1]["token"] == "a"


class TestCompletionResponseEdgeCases:
    def test_custom_request_id(self) -> None:
        events = [TokenEvent(token_id=1, text="x", finish_reason=FinishReason.STOP)]
        resp = build_completion_response(events, request_id="my-id")
        assert resp["id"] == "my-id"

    def test_length_finish_reason(self) -> None:
        events = [
            TokenEvent(token_id=1, text="x", finish_reason=FinishReason.LENGTH),
        ]
        resp = build_completion_response(events)
        assert resp["choices"][0]["finish_reason"] == "length"

    def test_concatenates_all_text(self) -> None:
        events = [TokenEvent(token_id=i, text=f"w{i}") for i in range(10)]
        events[-1].finish_reason = FinishReason.STOP
        resp = build_completion_response(events)
        expected = "".join(f"w{i}" for i in range(10))
        assert resp["choices"][0]["message"]["content"] == expected
