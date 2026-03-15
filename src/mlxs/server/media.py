"""Media processing for multimodal requests (FR12, §7.4).

Extracts images, audio, and video from OpenAI-format messages,
preprocesses them, and produces input_embeddings via the model's
prepare_inputs(). Pillow / soundfile are imported lazily.
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
                    media_item = _parse_data_url(url, media_type="image")
                    media_items.append(media_item)
                    text_parts.append("<image>")
                else:
                    raise InvalidPromptError(
                        f"Only base64 data URLs are supported for images, got: {url[:50]}..."
                    )
            elif part_type == "input_audio":
                audio_obj = part.get("input_audio", {})
                data_b64 = audio_obj.get("data", "")
                fmt = audio_obj.get("format", "wav")
                if not data_b64:
                    raise InvalidPromptError("input_audio missing 'data' field")
                data = base64.b64decode(data_b64)
                media_items.append(MediaItem(
                    media_type="audio", data=data, mime_type=f"audio/{fmt}",
                ))
                text_parts.append("<audio>")
            elif part_type == "video_url":
                video_url_obj = part.get("video_url", {})
                url = video_url_obj.get("url", "")
                if url.startswith("data:"):
                    media_item = _parse_data_url(url, media_type="video")
                    media_items.append(media_item)
                    text_parts.append("<video>")
                else:
                    raise InvalidPromptError(
                        f"Only base64 data URLs are supported for video, got: {url[:50]}..."
                    )
            else:
                raise InvalidPromptError(f"Unsupported content type: {part_type}")

        text_messages.append({**msg, "content": "\n".join(text_parts)})

    return text_messages, media_items


def _parse_data_url(url: str, *, media_type: str = "image") -> MediaItem:
    """Parse a data URL (``data:image/jpeg;base64,...``) into a MediaItem."""
    try:
        header, encoded = url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        data = base64.b64decode(encoded)
        return MediaItem(media_type=media_type, data=data, mime_type=mime_type)
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
    except ImportError as exc:
        raise InvalidPromptError(
            "Pillow is required for image processing. "
            "Install with: pip install mlxs[vision]"
        ) from exc
    except Exception as exc:
        raise InvalidPromptError(f"Failed to load image: {exc}") from exc


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

    if model_type in (
        "qwen2",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3",
        "qwen3_vl",
        "qwen3_moe",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_vl",
    ):
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


def process_audio_inputs(
    model: Any,
    audio_items: list[MediaItem],
    input_ids: mx.array,
) -> tuple[mx.array, mx.array | None, str | None]:
    """Preprocess audio items and produce input_embeddings.

    Args:
        model: Model with audio support (e.g., gemma3n multimodal).
        audio_items: MediaItem list with media_type="audio".
        input_ids: Token ids ``(1, T)`` with ``<audio>`` placeholder tokens.

    Returns:
        input_ids, input_embeddings, media_hash.
    """
    if not audio_items:
        return input_ids, None, None

    from mlxs.models.vision.audio_processing import load_audio, preprocess_audio_batch

    waveforms = [load_audio(item.data) for item in audio_items]
    audio_mel, audio_mel_mask = preprocess_audio_batch(waveforms)

    # Pass audio features through model.prepare_inputs if available
    if hasattr(model, "prepare_inputs"):
        input_ids_out, input_embeddings = model.prepare_inputs(
            input_ids,
            input_features=audio_mel,
            input_features_mask=audio_mel_mask,
        )
    else:
        input_ids_out = input_ids
        input_embeddings = None

    if input_embeddings is not None:
        input_embeddings = input_embeddings.squeeze(0)

    # Hash for cache keying
    raw_bytes = b"".join(item.data[:256] for item in audio_items)
    media_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]

    return input_ids_out, input_embeddings, media_hash


def process_video_inputs(
    model: Any,
    video_items: list[MediaItem],
    input_ids: mx.array,
    *,
    target_fps: float = 1.0,
    max_frames: int = 64,
    image_max_pixels: int | None = None,
) -> tuple[mx.array, mx.array | None, str | None]:
    """Preprocess video items and produce input_embeddings.

    Extracts frames, processes through the image pipeline, and runs
    model.prepare_inputs(). Currently processes the first video item only.

    Returns:
        input_ids, input_embeddings, media_hash.
    """
    if not video_items:
        return input_ids, None, None

    import os
    import tempfile

    from mlxs.models.vision.video_processing import (
        extract_video_frames,
        preprocess_video_qwen_vl,
        preprocess_video_standard,
    )

    item = video_items[0]  # Process first video

    # Write to temp file for av/cv2 to read
    suffix = "." + item.mime_type.split("/")[-1] if "/" in item.mime_type else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(item.data)
        tmp_path = tmp.name

    try:
        frames = extract_video_frames(
            tmp_path, target_fps=target_fps, max_frames=max_frames,
        )
    finally:
        os.unlink(tmp_path)

    if not frames:
        return input_ids, None, None

    model_type = getattr(model, "model_type", "")

    if model_type in (
        "qwen2",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3",
        "qwen3_vl",
        "qwen3_moe",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_vl",
    ):
        vision_cfg = {"patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2}
        pixel_values, grid_thw = preprocess_video_qwen_vl(
            frames, vision_cfg, max_pixels=image_max_pixels,
        )
        if model_type in ("qwen3_5", "qwen3_5_vl"):
            extra_kwargs = {"video_grid_thw": grid_thw}
        else:
            extra_kwargs = {"image_grid_thw": grid_thw}
    else:
        pixel_values = preprocess_video_standard(frames)
        extra_kwargs = {}

    if hasattr(model, "prepare_inputs"):
        if model_type in ("qwen3_5", "qwen3_5_vl"):
            input_ids_out, input_embeddings = model.prepare_inputs(
                input_ids,
                video_pixel_values=pixel_values,
                **extra_kwargs,
            )
        else:
            input_ids_out, input_embeddings = model.prepare_inputs(
                input_ids,
                pixel_values=pixel_values,
                **extra_kwargs,
            )
    else:
        input_ids_out = input_ids
        input_embeddings = None

    if input_embeddings is not None:
        input_embeddings = input_embeddings.squeeze(0)

    media_hash = hashlib.sha256(item.data[:512]).hexdigest()[:16]
    return input_ids_out, input_embeddings, media_hash
