"""Chat completions endpoint — OpenAI-compatible (§6.7, FR8, AC6).

POST /v1/chat/completions — supports both streaming (SSE) and
non-streaming responses.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mlxs._types import GenerateOptions, TokenEvent
from mlxs.server.sse import build_completion_response, token_events_to_sse

logger = logging.getLogger(__name__)


async def chat_completions(request: Request) -> Response:
    """POST /v1/chat/completions — chat completion endpoint.

    Expects OpenAI-compatible request body. Delegates to the generate
    function via the app's dependency container.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON body"}}, status_code=400)

    # Extract parameters
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(
            {"error": {"message": "messages is required and must not be empty"}},
            status_code=400,
        )

    stream = body.get("stream", False)
    model_id = body.get("model", "mlxs")

    # Build generate options from request
    options = _build_options(body)

    # Get dependencies from app state
    deps = request.app.state.deps

    try:
        # Apply chat template to get prompt
        prompt = _apply_chat_template(deps.tokenizer, messages)
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": f"Failed to apply chat template: {exc}"}},
            status_code=400,
        )

    if stream:
        return await _stream_response(deps, prompt, options, model_id)
    return await _non_stream_response(deps, prompt, options, model_id)


def _build_options(body: dict[str, Any]) -> GenerateOptions:
    """Build GenerateOptions from an OpenAI-style request body."""
    return GenerateOptions(
        max_tokens=body.get("max_tokens", 512),
        temperature=body.get("temperature", 1.0),
        top_p=body.get("top_p", 1.0),
        top_k=body.get("top_k", 0),
        min_p=body.get("min_p", 0.0),
        seed=body.get("seed"),
        stop_sequences=tuple(body.get("stop", ())),
        stream=body.get("stream", False),
        logprobs=body.get("logprobs", False),
        top_logprobs=body.get("top_logprobs", 0),
    )


def _apply_chat_template(tokenizer: Any, messages: list[dict]) -> str:
    """Apply chat template to messages, returning the formatted prompt string."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: concatenate message contents
    return "\n".join(m.get("content", "") for m in messages)


async def _stream_response(
    deps: Any,
    prompt: str,
    options: GenerateOptions,
    model_id: str,
) -> Response:
    """Handle streaming (SSE) response."""
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        # Run generate in thread to avoid blocking event loop (Plan §A1)
        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            None,
            lambda: list(deps.generate_fn(deps.model, deps.tokenizer, prompt, options)),
        )

        for sse_chunk in token_events_to_sse(iter(events), model_id=model_id):
            yield sse_chunk

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


async def _non_stream_response(
    deps: Any,
    prompt: str,
    options: GenerateOptions,
    model_id: str,
) -> JSONResponse:
    """Handle non-streaming response."""
    loop = asyncio.get_event_loop()
    events: list[TokenEvent] = await loop.run_in_executor(
        None,
        lambda: list(deps.generate_fn(deps.model, deps.tokenizer, prompt, options)),
    )

    response = build_completion_response(events, model_id=model_id)
    return JSONResponse(response)
