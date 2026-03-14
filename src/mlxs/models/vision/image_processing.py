"""Image preprocessing for multimodal models (§7.4, FR12).

Per-family preprocessing functions that convert PIL images into the
tensor formats expected by each vision encoder. Pillow is imported
lazily — only when actually called.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from mlxs._errors import InvalidPromptError

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Qwen2-VL defaults
_QWEN_VL_MIN_PIXELS = 3136     # 4 * 28 * 28
_QWEN_VL_MAX_PIXELS = 12845056  # ~3584 * 3584


def _ensure_pillow() -> Any:
    """Lazily import PIL, raise clear error if missing."""
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise InvalidPromptError(
            "Pillow is required for image processing. "
            "Install with: pip install mlxs[vision]"
        )


def _smart_resize(
    height: int, width: int, factor: int = 28,
    min_pixels: int = _QWEN_VL_MIN_PIXELS, max_pixels: int = _QWEN_VL_MAX_PIXELS,
) -> tuple[int, int]:
    """Resize dimensions preserving aspect ratio, constraining to min/max pixels.

    Rounds to nearest multiple of ``factor`` (patch_size * spatial_merge_size).
    """
    if max(height, width) / min(height, width) > 200:
        raise InvalidPromptError(
            f"Image aspect ratio too extreme: {width}x{height}"
        )

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return max(h_bar, factor), max(w_bar, factor)


def _normalize_image(
    img_array: mx.array,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> mx.array:
    """Normalize image array (C, H, W) with mean/std."""
    mean_arr = mx.array(mean, dtype=mx.float32).reshape(3, 1, 1)
    std_arr = mx.array(std, dtype=mx.float32).reshape(3, 1, 1)
    return (img_array - mean_arr) / std_arr


def preprocess_qwen_vl(
    images: list[Any],  # list[PIL.Image.Image]
    vision_config: dict[str, Any] | Any,
    *,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Preprocess images for Qwen2-VL / Qwen3-VL.

    Args:
        images: PIL images (RGB).
        vision_config: VisionConfig dataclass or dict with vision params.
        max_pixels: Override max total pixels per image.
        min_pixels: Override min total pixels per image.

    Returns:
        pixel_values: ``(N_patches_total, C * temporal_patch_size, patch_H, patch_W)``
        image_grid_thw: ``(N_images, 3)`` — ``[temporal, height, width]`` per image.
    """
    Image = _ensure_pillow()

    # Extract config values
    if isinstance(vision_config, dict):
        patch_size = vision_config.get("patch_size", 14)
        temporal_patch_size = vision_config.get("temporal_patch_size", 2)
        spatial_merge_size = vision_config.get("spatial_merge_size", 2)
    else:
        patch_size = getattr(vision_config, "patch_size", 14)
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)

    factor = patch_size * spatial_merge_size  # typically 28
    min_px = min_pixels or _QWEN_VL_MIN_PIXELS
    max_px = max_pixels or _QWEN_VL_MAX_PIXELS

    all_patches = []
    grid_thw = []

    for img in images:
        if not isinstance(img, Image.Image):
            raise InvalidPromptError(f"Expected PIL Image, got {type(img)}")

        img = img.convert("RGB")
        w, h = img.size

        new_h, new_w = _smart_resize(h, w, factor=factor, min_pixels=min_px, max_pixels=max_px)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Convert to float array (C, H, W) in [0, 1]
        import numpy as np
        img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        img_arr = mx.array(img_np).transpose(2, 0, 1)      # (3, H, W)

        # Normalize
        img_arr = _normalize_image(img_arr)

        # For images (not video), temporal=1 but we need temporal_patch_size frames
        # Duplicate frame to fill temporal dimension
        # Shape: (C, temporal_patch_size, H, W)
        img_arr = mx.stack([img_arr] * temporal_patch_size, axis=1)

        # Patchify: reshape into patches
        c = 3
        t = temporal_patch_size
        grid_h = new_h // patch_size
        grid_w = new_w // patch_size
        num_patches = grid_h * grid_w

        # (C, T, H, W) -> (num_patches, C * T, patch_H, patch_W)
        patches = img_arr.reshape(c, t, grid_h, patch_size, grid_w, patch_size)
        patches = patches.transpose(2, 4, 1, 0, 3, 5)  # (grid_h, grid_w, T, C, pH, pW)
        patches = patches.reshape(num_patches, c * t, patch_size, patch_size)

        all_patches.append(patches)
        grid_thw.append([1, grid_h, grid_w])  # temporal=1 for images

    pixel_values = mx.concatenate(all_patches, axis=0)
    image_grid_thw = mx.array(grid_thw, dtype=mx.int32)

    return pixel_values, image_grid_thw


def preprocess_pixtral(
    images: list[Any],  # list[PIL.Image.Image]
    vision_config: dict[str, Any] | Any = None,
    *,
    max_pixels: int | None = None,
) -> tuple[mx.array, list[tuple[int, int]]]:
    """Preprocess images for Pixtral / Mistral3.

    Returns:
        pixel_values: ``(N, C, H, W)`` — each image resized to patch-aligned dims.
        image_sizes: list of ``(H, W)`` per image (actual resolution).
    """
    Image = _ensure_pillow()
    import numpy as np

    if isinstance(vision_config, dict):
        patch_size = vision_config.get("patch_size", 14)
        image_size = vision_config.get("image_size", 336)
    elif vision_config is not None:
        patch_size = getattr(vision_config, "patch_size", 14)
        image_size = getattr(vision_config, "image_size", 336)
    else:
        patch_size = 14
        image_size = 336

    max_px = max_pixels or (image_size * image_size * 4)

    processed = []
    sizes = []

    for img in images:
        if not isinstance(img, Image.Image):
            raise InvalidPromptError(f"Expected PIL Image, got {type(img)}")

        img = img.convert("RGB")
        w, h = img.size

        # Resize to fit within max_pixels, keep aspect ratio, align to patch_size
        if h * w > max_px:
            scale = math.sqrt(max_px / (h * w))
            h = int(h * scale)
            w = int(w * scale)

        # Align to patch_size
        h = max(patch_size, (h // patch_size) * patch_size)
        w = max(patch_size, (w // patch_size) * patch_size)

        img = img.resize((w, h), Image.BICUBIC)
        sizes.append((h, w))

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_arr = mx.array(img_np).transpose(2, 0, 1)  # (C, H, W)
        img_arr = _normalize_image(img_arr, mean=CLIP_MEAN, std=CLIP_STD)
        processed.append(img_arr)

    # Pad to max H, W for batching
    max_h = max(s[0] for s in sizes)
    max_w = max(s[1] for s in sizes)
    padded = []
    for arr in processed:
        c, h, w = arr.shape
        if h < max_h or w < max_w:
            padded_arr = mx.zeros((c, max_h, max_w), dtype=arr.dtype)
            padded_arr[:, :h, :w] = arr
            padded.append(padded_arr)
        else:
            padded.append(arr)

    pixel_values = mx.stack(padded, axis=0)  # (N, C, H, W)
    return pixel_values, sizes


def preprocess_standard(
    images: list[Any],  # list[PIL.Image.Image]
    image_size: int = 384,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> mx.array:
    """Standard preprocessing: resize + center crop + normalize.

    Returns: ``(N, C, H, W)`` tensor.
    """
    Image = _ensure_pillow()
    import numpy as np

    processed = []
    for img in images:
        if not isinstance(img, Image.Image):
            raise InvalidPromptError(f"Expected PIL Image, got {type(img)}")

        img = img.convert("RGB")
        # Resize shortest side to image_size, then center crop
        w, h = img.size
        scale = image_size / min(w, h)
        new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Center crop
        left = (new_w - image_size) // 2
        top = (new_h - image_size) // 2
        img = img.crop((left, top, left + image_size, top + image_size))

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_arr = mx.array(img_np).transpose(2, 0, 1)  # (C, H, W)
        img_arr = _normalize_image(img_arr, mean=mean, std=std)
        processed.append(img_arr)

    return mx.stack(processed, axis=0)
