"""Media processing for multimodal requests (FR12, §7.4).

Extracts images from OpenAI-format messages, preprocesses them, and
produces input_embeddings via the model's prepare_inputs(). Pillow
is imported lazily — only when images are present.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import mlx.core as mx

from mlxs._errors import InvalidPromptError


@dataclass
class MediaItem:
    """A single media resource extracted from a request."""

    media_type: str  # "image", "audio", "video"
    data: bytes      # raw bytes (after base64 decode)
    mime_type: str   # "image/jpeg", "image/png", etc.


def extract_media_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[MediaItem]]:
    """Extract media items from OpenAI-format messages.

    Converts messages with content arrays into messages with string content
    (text only) plus a separate list of MediaItem objects.

    Returns:
        text_messages: Messages with string content (placeholder ``<image>`` inserted).
        media_items: Extracted MediaItem list.

    Raises:
        InvalidPromptError: On invalid or unsupported media format.
    """
    text_messages: list[dict[str, Any]] = []
    media_items: list[MediaItem] = []

    for msg in messages:
        content = msg.get("content")
        if content is None or isinstance(content, str):
            text_messages.append(msg)
            continue

        if not isinstance(content, list):
            text_messages.append(msg)
            continue

        # Content is a list of parts
        text_parts: list[str] = []
        for part in content:
            part_type = part.get("type", "")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif part_type == "image_url":
                image_url_obj = part.get("image_url", {})
                url = image_url_obj.get("url", "")
                if url.startswith("data:"):
                    media_item = _parse_data_url(url)
                    media_items.append(media_item)
                    text_parts.append("<image>")
                else:
                    raise InvalidPromptError(
                        f"Only base64 data URLs are supported for images, got: {url[:50]}..."
                    )
            else:
                if part_type in ("audio", "video"):
                    raise InvalidPromptError(f"{part_type.capitalize()} not yet supported")
                raise InvalidPromptError(f"Unsupported content type: {part_type}")

        text_messages.append({**msg, "content": "\n".join(text_parts)})

    return text_messages, media_items


def _parse_data_url(url: str) -> MediaItem:
    """Parse a data URL (``data:image/jpeg;base64,...``) into a MediaItem."""
    try:
        header, encoded = url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        data = base64.b64decode(encoded)
        return MediaItem(media_type="image", data=data, mime_type=mime_type)
    except (ValueError, IndexError) as exc:
        raise InvalidPromptError(f"Invalid data URL format: {exc}") from exc


def load_image(item: MediaItem) -> Any:
    """Load a MediaItem as a PIL Image.

    Raises:
        InvalidPromptError: If the image is corrupt or Pillow is missing.
    """
    try:
        from PIL import Image

        return Image.open(BytesIO(item.data)).convert("RGB")
    except ImportError:
        raise InvalidPromptError(
            "Pillow is required for image processing. "
            "Install with: pip install mlxs[vision]"
        )
    except Exception as exc:
        raise InvalidPromptError(f"Failed to load image: {exc}")


def process_media_inputs(
    model: Any,
    images: list[Any],  # list[PIL.Image.Image]
    input_ids: mx.array,
    *,
    image_max_pixels: int | None = None,
    image_min_pixels: int | None = None,
) -> tuple[mx.array, mx.array | None, str | None]:
    """Preprocess images and run model.prepare_inputs().

    Args:
        model: Model with ``prepare_inputs`` (MultimodalModelProtocol).
        images: PIL images to encode.
        input_ids: Token ids ``(1, T)`` with placeholder tokens.
        image_max_pixels: Override max pixels per image.
        image_min_pixels: Override min pixels per image.

    Returns:
        input_ids: Unchanged ``(1, T)``.
        input_embeddings: ``(T, D)`` merged embeddings, or None if no images.
        media_hash: SHA-256 hash prefix of pixel values for cache keying.
    """
    if not images:
        return input_ids, None, None

    model_type = getattr(model, "model_type", "")
    pixel_values, extra_kwargs = _preprocess_images(
        model_type, images,
        max_pixels=image_max_pixels, min_pixels=image_min_pixels,
    )

    input_ids_out, input_embeddings = model.prepare_inputs(
        input_ids, pixel_values=pixel_values, **extra_kwargs,
    )

    if input_embeddings is not None:
        input_embeddings = input_embeddings.squeeze(0)  # (1, T, D) -> (T, D)

    media_hash = hashlib.sha256(
        pixel_values.reshape(-1).astype(mx.float32).tolist().__repr__().encode()
    ).hexdigest()[:16]

    return input_ids_out, input_embeddings, media_hash


def _preprocess_images(
    model_type: str,
    images: list[Any],
    *,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> tuple[mx.array, dict[str, Any]]:
    """Route to per-family preprocessing.

    Returns: ``(pixel_values, extra_kwargs for prepare_inputs)``
    """
    from mlxs.models.vision.image_processing import (
        preprocess_pixtral,
        preprocess_qwen_vl,
        preprocess_standard,
    )

    if model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"):
        vision_cfg = {"patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2}
        pv, grid = preprocess_qwen_vl(
            images, vision_cfg, max_pixels=max_pixels, min_pixels=min_pixels,
        )
        return pv, {"image_grid_thw": grid}
    elif model_type in ("pixtral", "mistral3"):
        pv, sizes = preprocess_pixtral(images, max_pixels=max_pixels)
        return pv, {"image_sizes": sizes}
    else:
        pv = preprocess_standard(images)
        return pv, {}
