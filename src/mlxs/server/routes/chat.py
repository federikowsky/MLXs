"""Compatibility chat-completions route delegating to the Layer 4 API surface."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from mlxs._types import GenerateOptions
from mlxs.product_surfaces.compat_openai import (
    build_multimodal_embeddings,
    build_openai_options,
    handle_chat_completions,
)
from mlxs.server.media import (
    extract_media_from_messages,
    load_image,
    process_media_inputs,
    process_video_inputs,
)
from mlxs.server.sse import build_completion_response, token_events_to_sse


async def chat_completions(request: Request) -> Response:
    """Compatibility wrapper for the Layer 4 OpenAI-compatible HTTP surface."""
    return await handle_chat_completions(
        request,
        extract_media_from_messages_fn=extract_media_from_messages,
        load_image_fn=load_image,
        process_media_inputs_fn=process_media_inputs,
        process_video_inputs_fn=process_video_inputs,
        build_completion_response_fn=build_completion_response,
        token_events_to_sse_fn=token_events_to_sse,
    )


def _build_options(body: dict[str, Any]) -> GenerateOptions:
    return build_openai_options(body)


def _build_multimodal_embeddings(
    deps: Any,
    prompt: str,
    media_items: list[Any],
) -> Any:
    return build_multimodal_embeddings(
        deps,
        prompt,
        media_items,
        load_image_fn=load_image,
        process_media_inputs_fn=process_media_inputs,
        process_video_inputs_fn=process_video_inputs,
    )
