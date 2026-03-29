"""Chat completions endpoint — OpenAI-compatible (§6.7, FR8, AC6).

POST /v1/chat/completions — supports both streaming (SSE) and
non-streaming responses.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mlxs._errors import InvalidPromptError
from mlxs._types import GenerateOptions, TokenEvent
from mlxs.chat.template import build_prompt_str
from mlxs.server.media import (
    extract_media_from_messages,
    load_image,
    process_media_inputs,
    process_video_inputs,
)
from mlxs.server.sse import build_completion_response, token_event_to_openai_stream_json

logger = logging.getLogger(__name__)

_VIDEO_SERVING_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
_STREAM_END = object()


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
        prompt = build_prompt_str(deps.tokenizer, text_messages)
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": f"Failed to apply chat template: {exc}"}},
            status_code=400,
        )

    # Process media if present (§7.4)
    input_embeddings = None
    if media_items:
        try:
            input_embeddings = _build_multimodal_embeddings(
                deps,
                prompt,
                media_items,
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


def _build_multimodal_embeddings(
    deps: Any,
    prompt: str,
    media_items: list[Any],
) -> Any:
    """Build route-level multimodal embeddings within the supported serving scope."""
    image_items = [item for item in media_items if item.media_type == "image"]
    audio_items = [item for item in media_items if item.media_type == "audio"]
    video_items = [item for item in media_items if item.media_type == "video"]

    if audio_items:
        raise InvalidPromptError(
            "Audio inputs are not supported via /v1/chat/completions "
            "in the current production scope."
        )

    if image_items and video_items:
        raise InvalidPromptError(
            "Mixed image and video requests are not supported via "
            "/v1/chat/completions in the current production scope."
        )

    prompt_tokens = deps.tokenizer.encode(prompt)
    import mlx.core as mx

    input_ids = mx.array(prompt_tokens)[None]  # (1, T)

    if image_items:
        max_images = deps.config.model.max_images_per_request
        if len(image_items) > max_images:
            raise InvalidPromptError(f"Too many images: {len(image_items)} > {max_images}")

        images = [load_image(item) for item in image_items]
        _, input_embeddings, _ = process_media_inputs(
            deps.model,
            images,
            input_ids,
            image_max_pixels=deps.config.model.image_max_pixels,
            image_min_pixels=deps.config.model.image_min_pixels,
        )
        if input_embeddings is None:
            raise InvalidPromptError(
                "Image inputs are not supported for the loaded model configuration."
            )
        return input_embeddings

    if video_items:
        if len(video_items) > 1:
            raise InvalidPromptError(
                "At most one video per request is supported via /v1/chat/completions."
            )

        model_type = getattr(deps.model, "model_type", "")
        if model_type not in _VIDEO_SERVING_MODEL_TYPES:
            supported = ", ".join(sorted(_VIDEO_SERVING_MODEL_TYPES))
            raise InvalidPromptError(
                "Video inputs via /v1/chat/completions are supported only for "
                f"{supported}; got {model_type or 'unknown'}."
            )

        _, input_embeddings, _ = process_video_inputs(
            deps.model,
            video_items,
            input_ids,
            image_max_pixels=deps.config.model.image_max_pixels,
        )
        if input_embeddings is None:
            raise InvalidPromptError("Failed to derive video embeddings from the request.")
        return input_embeddings

    return None


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
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()
        error: list[Exception | None] = [None]

        def producer() -> None:
            try:
                for event in deps.generate_fn(
                    deps.model,
                    deps.tokenizer,
                    prompt,
                    options,
                    input_embeddings=input_embeddings,
                ):
                    fut = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
                    fut.result()
            except Exception as exc:
                error[0] = exc
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_STREAM_END), loop).result()

        threading.Thread(target=producer, daemon=True).start()

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        while True:
            item = await queue.get()
            if item is _STREAM_END:
                if error[0] is not None:
                    raise error[0]
                yield "[DONE]"
                return
            assert isinstance(item, TokenEvent)
            yield token_event_to_openai_stream_json(
                item,
                model_id=model_id,
                request_id=request_id,
                created=created,
            )

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
