"""Layer 4 OpenAI-compatible HTTP/SSE surface adapters."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
import time
import uuid
from typing import Any

from mlxs._errors import CapacityExceededError, InvalidPromptError, MLXsError, RequestTimeoutError
from mlxs._types import GenerateOptions, TokenEvent
from mlxs.chat.template import build_prompt_str
from mlxs.product_surfaces.observability import record_counter

logger = logging.getLogger(__name__)

_VIDEO_SERVING_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})


def build_openai_options(body: dict[str, Any]) -> GenerateOptions:
    """Build Layer 4 compatibility options from an OpenAI-style request body."""
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


def build_multimodal_embeddings(
    runtime: Any,
    prompt: str,
    media_items: list[Any],
    *,
    load_image_fn: Callable[[Any], Any],
    process_media_inputs_fn: Callable[..., Any],
    process_video_inputs_fn: Callable[..., Any],
) -> Any:
    """Build Layer 4 multimodal embeddings within the serving scope."""
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

    prompt_tokens = runtime.tokenizer.encode(prompt)
    import mlx.core as mx

    input_ids = mx.array(prompt_tokens)[None]

    if image_items:
        max_images = runtime.config.model.max_images_per_request
        if len(image_items) > max_images:
            raise InvalidPromptError(f"Too many images: {len(image_items)} > {max_images}")

        images = [load_image_fn(item) for item in image_items]
        _, input_embeddings, _ = process_media_inputs_fn(
            runtime.model,
            images,
            input_ids,
            image_max_pixels=runtime.config.model.image_max_pixels,
            image_min_pixels=runtime.config.model.image_min_pixels,
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

        model_type = getattr(runtime.model, "model_type", "")
        if model_type not in _VIDEO_SERVING_MODEL_TYPES:
            supported = ", ".join(sorted(_VIDEO_SERVING_MODEL_TYPES))
            raise InvalidPromptError(
                "Video inputs via /v1/chat/completions are supported only for "
                f"{supported}; got {model_type or 'unknown'}."
            )

        _, input_embeddings, _ = process_video_inputs_fn(
            runtime.model,
            video_items,
            input_ids,
            image_max_pixels=runtime.config.model.image_max_pixels,
        )
        if input_embeddings is None:
            raise InvalidPromptError("Failed to derive video embeddings from the request.")
        return input_embeddings

    return None


def _surface_runtime(request: Any) -> Any:
    return getattr(request.app.state, "runtime", getattr(request.app.state, "deps", None))


def _json_error(message: str, status_code: int) -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse({"error": {"message": message}}, status_code=status_code)


async def _release_request_slot(runtime: Any) -> None:
    request_queue = getattr(runtime, "request_queue", None)
    if request_queue is None:
        return
    try:
        await request_queue.get()
    except Exception:
        pass


async def _execute_generation(
    runtime: Any,
    prompt: str,
    options: GenerateOptions,
    *,
    input_embeddings: Any,
) -> list[TokenEvent]:
    request_queue = getattr(runtime, "request_queue", None)
    timeout = getattr(getattr(getattr(runtime, "config", None), "server", None), "request_timeout", None)
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout
    if request_queue is not None:
        queue_timeout = None if deadline is None else max(0.0, deadline - loop.time())
        await request_queue.put({"prompt": prompt}, timeout=queue_timeout)
    record_counter(runtime, "product_requests_total", 1.0, surface="http")

    async def _run() -> list[TokenEvent]:
        batch_host = getattr(runtime, "batch_host", None)
        if batch_host is not None and input_embeddings is None:
            return await batch_host.execute(prompt, options)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: list(
                runtime.generate_fn(
                    runtime.model,
                    runtime.tokenizer,
                    prompt,
                    options,
                    input_embeddings=input_embeddings,
                )
            ),
        )

    try:
        if timeout is None:
            events = await _run()
        else:
            remaining = max(0.0, deadline - loop.time())
            events = await asyncio.wait_for(_run(), timeout=remaining)
        record_counter(runtime, "product_requests_completed_total", 1.0, surface="http")
        return events
    except asyncio.TimeoutError as exc:
        record_counter(runtime, "product_requests_timeout_total", 1.0, surface="http")
        raise RequestTimeoutError(f"Request timed out after {timeout} seconds.") from exc
    finally:
        await _release_request_slot(runtime)


async def _stream_generation(
    runtime: Any,
    prompt: str,
    options: GenerateOptions,
):
    """Stream generation through the Layer 4 product surface."""
    request_queue = getattr(runtime, "request_queue", None)
    timeout = getattr(getattr(getattr(runtime, "config", None), "server", None), "request_timeout", None)
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout
    if request_queue is not None:
        queue_timeout = None if deadline is None else max(0.0, deadline - loop.time())
        await request_queue.put({"prompt": prompt}, timeout=queue_timeout)
    record_counter(runtime, "product_requests_total", 1.0, surface="http")

    batch_host = getattr(runtime, "batch_host", None)
    if batch_host is None:
        events = await _execute_generation(
            runtime,
            prompt,
            options,
            input_embeddings=None,
        )
        for event in events:
            yield event
        return

    try:
        async for event in batch_host.stream_execute(prompt, options):
            if timeout is not None and loop.time() >= deadline:
                raise RequestTimeoutError(f"Request timed out after {timeout} seconds.")
            yield event
        record_counter(runtime, "product_requests_completed_total", 1.0, surface="http")
    finally:
        await _release_request_slot(runtime)


async def handle_chat_completions(
    request: Any,
    *,
    extract_media_from_messages_fn: Callable[[list[Any]], Any],
    load_image_fn: Callable[[Any], Any],
    process_media_inputs_fn: Callable[..., Any],
    process_video_inputs_fn: Callable[..., Any],
    build_completion_response_fn: Callable[..., dict[str, Any]],
    token_events_to_sse_fn: Callable[..., Any],
) -> Any:
    """Handle the OpenAI-compatible chat-completions surface as Layer 4 logic."""
    try:
        body = await request.json()
    except Exception:
        return _json_error("Invalid JSON body", 400)

    messages = body.get("messages", [])
    if not messages:
        return _json_error("messages is required and must not be empty", 400)

    stream = body.get("stream", False)
    model_id = body.get("model", "mlxs")
    runtime = _surface_runtime(request)
    options = build_openai_options(body)

    try:
        text_messages, media_items = extract_media_from_messages_fn(messages)
        prompt = build_prompt_str(runtime.tokenizer, text_messages)
        input_embeddings = None
        if media_items:
            input_embeddings = build_multimodal_embeddings(
                runtime,
                prompt,
                media_items,
                load_image_fn=load_image_fn,
                process_media_inputs_fn=process_media_inputs_fn,
                process_video_inputs_fn=process_video_inputs_fn,
            )
        if stream and input_embeddings is None:
            events = None
        else:
            events = await _execute_generation(
                runtime,
                prompt,
                options,
                input_embeddings=input_embeddings,
            )
    except CapacityExceededError as exc:
        record_counter(runtime, "product_requests_rejected_total", 1.0, surface="http")
        return _json_error(str(exc), exc.status_hint)
    except InvalidPromptError as exc:
        return _json_error(str(exc), exc.status_hint)
    except RequestTimeoutError as exc:
        return _json_error(str(exc), exc.status_hint)
    except MLXsError as exc:
        return _json_error(str(exc), exc.status_hint)
    except Exception as exc:
        logger.exception("Unhandled product-surface error")
        return _json_error(f"Error: {exc}", 500)

    if stream:
        from starlette.responses import StreamingResponse
        from mlxs.server.sse import build_sse_chunk

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async def event_generator():
            async for event in _stream_generation(runtime, prompt, options):
                yield build_sse_chunk(
                    event,
                    model_id=model_id,
                    request_id=request_id,
                    created=created,
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    from starlette.responses import JSONResponse

    assert events is not None
    response = build_completion_response_fn(events, model_id=model_id)
    return JSONResponse(response)
