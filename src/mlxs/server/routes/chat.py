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

from mlxs._errors import InvalidPromptError
from mlxs._types import GenerateOptions, TokenEvent
from mlxs.server.media import (
    extract_media_from_messages,
    load_image,
    process_media_inputs,
)
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

    # Extract media from messages (§7.4, FR12)
    try:
        text_messages, media_items = extract_media_from_messages(messages)
    except InvalidPromptError as exc:
        return JSONResponse(
            {"error": {"message": str(exc)}}, status_code=400,
        )

    try:
        prompt = _apply_chat_template(deps.tokenizer, text_messages)
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": f"Failed to apply chat template: {exc}"}},
            status_code=400,
        )

    # Process media if present (§7.4)
    input_embeddings = None
    if media_items:
        try:
            image_items = [m for m in media_items if m.media_type == "image"]
            max_images = deps.config.model.max_images_per_request
            if len(image_items) > max_images:
                msg = f"Too many images: {len(image_items)} > {max_images}"
                return JSONResponse(
                    {"error": {"message": msg}}, status_code=400,
                )
            images = [load_image(item) for item in image_items]
            prompt_tokens = deps.tokenizer.encode(prompt)
            import mlx.core as mx
            input_ids = mx.array(prompt_tokens)[None]  # (1, T)
            _, input_embeddings, _ = process_media_inputs(
                deps.model, images, input_ids,
                image_max_pixels=deps.config.model.image_max_pixels,
                image_min_pixels=deps.config.model.image_min_pixels,
            )
        except InvalidPromptError as exc:
            return JSONResponse(
                {"error": {"message": str(exc)}}, status_code=400,
            )

    if stream:
        return await _stream_response(deps, prompt, options, model_id, input_embeddings)
    return await _non_stream_response(deps, prompt, options, model_id, input_embeddings)


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
    input_embeddings: Any = None,
) -> Response:
    """Handle streaming (SSE) response."""
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            None,
            lambda: list(deps.generate_fn(
                deps.model, deps.tokenizer, prompt, options,
                input_embeddings=input_embeddings,
            )),
        )

        for sse_chunk in token_events_to_sse(iter(events), model_id=model_id):
            yield sse_chunk

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


async def _non_stream_response(
    deps: Any,
    prompt: str,
    options: GenerateOptions,
    model_id: str,
    input_embeddings: Any = None,
) -> JSONResponse:
    """Handle non-streaming response."""
    loop = asyncio.get_event_loop()
    events: list[TokenEvent] = await loop.run_in_executor(
        None,
        lambda: list(deps.generate_fn(
            deps.model, deps.tokenizer, prompt, options,
            input_embeddings=input_embeddings,
        )),
    )

    response = build_completion_response(events, model_id=model_id)
    return JSONResponse(response)
