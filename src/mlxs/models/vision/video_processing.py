"""Video preprocessing: extract frames and process through image pipeline (§7.4, FR12).

Extracts frames from video at a configurable FPS and feeds them through the
existing image preprocessing pipeline. Supports Qwen-VL temporal encoding
(temporal_patch_size grouping) and standard frame-level encoding.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from mlxs._errors import InvalidPromptError
from mlxs.models.vision.image_processing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _ensure_pillow,
    _normalize_image,
    _smart_resize,
)


def _extract_frames_av(
    video_path: str,
    *,
    target_fps: float = 1.0,
    max_frames: int = 64,
) -> list[Any]:
    """Extract frames from video using av (PyAV).

    Returns list of PIL Images.
    """
    try:
        import av
    except ImportError:
        raise InvalidPromptError(
            "PyAV is required for video processing. "
            "Install with: pip install av"
        )

    Image = _ensure_pillow()

    container = av.open(video_path)
    stream = container.streams.video[0]

    video_fps = float(stream.average_rate or stream.guessed_rate or 30)
    duration = float(stream.duration * stream.time_base) if stream.duration else 0
    total_frames = int(duration * video_fps) if duration > 0 else 0

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / target_fps))
    expected_frames = min(max_frames, max(1, total_frames // frame_interval))

    frames: list[Any] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % frame_interval != 0:
            continue
        img = frame.to_image().convert("RGB")
        frames.append(img)
        if len(frames) >= max_frames:
            break

    container.close()
    return frames


def _extract_frames_cv2(
    video_path: str,
    *,
    target_fps: float = 1.0,
    max_frames: int = 64,
) -> list[Any]:
    """Extract frames using OpenCV (fallback if PyAV not available).

    Returns list of PIL Images.
    """
    try:
        import cv2
    except ImportError:
        raise InvalidPromptError(
            "Either PyAV or OpenCV is required for video processing. "
            "Install with: pip install av  OR  pip install opencv-python"
        )

    Image = _ensure_pillow()
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InvalidPromptError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / target_fps))

    frames: list[Any] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            if len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    return frames


def extract_video_frames(
    video_path: str,
    *,
    target_fps: float = 1.0,
    max_frames: int = 64,
) -> list[Any]:
    """Extract frames from video at target FPS.

    Tries PyAV first, falls back to OpenCV.

    Args:
        video_path: Path to video file.
        target_fps: Target frames per second to extract.
        max_frames: Maximum number of frames to return.

    Returns:
        List of PIL Images (RGB).
    """
    try:
        return _extract_frames_av(
            video_path, target_fps=target_fps, max_frames=max_frames
        )
    except InvalidPromptError:
        return _extract_frames_cv2(
            video_path, target_fps=target_fps, max_frames=max_frames
        )


def preprocess_video_qwen_vl(
    frames: list[Any],  # list[PIL.Image.Image]
    vision_config: dict[str, Any] | Any,
    *,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Preprocess video frames for Qwen2-VL / Qwen3-VL temporal encoding.

    Groups frames into temporal_patch_size chunks and creates 3D patches.
    Falls back to image preprocessing per frame when frame count is small.

    Returns:
        pixel_values: ``(N_patches_total, C * temporal_patch_size, patch_H, patch_W)``
        grid_thw: ``(1, 3)`` — ``[temporal_groups, grid_h, grid_w]``
    """
    Image = _ensure_pillow()
    import numpy as np

    if isinstance(vision_config, dict):
        patch_size = vision_config.get("patch_size", 14)
        temporal_patch_size = vision_config.get("temporal_patch_size", 2)
        spatial_merge_size = vision_config.get("spatial_merge_size", 2)
    else:
        patch_size = getattr(vision_config, "patch_size", 14)
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)

    factor = patch_size * spatial_merge_size
    min_px = min_pixels or 3136
    max_px = max_pixels or 12845056

    if not frames:
        raise InvalidPromptError("No frames extracted from video")

    # Pad frame count to multiple of temporal_patch_size
    n_frames = len(frames)
    n_temporal_groups = max(1, (n_frames + temporal_patch_size - 1) // temporal_patch_size)
    target_n = n_temporal_groups * temporal_patch_size

    # Duplicate last frame if needed
    while len(frames) < target_n:
        frames.append(frames[-1])

    # Resize all frames to same dimensions (use first frame's aspect ratio)
    first = frames[0]
    w0, h0 = first.size
    new_h, new_w = _smart_resize(h0, w0, factor=factor, min_pixels=min_px, max_pixels=max_px)

    grid_h = new_h // patch_size
    grid_w = new_w // patch_size

    # Process all frames
    frame_arrays = []
    for img in frames:
        if not isinstance(img, Image.Image):
            raise InvalidPromptError(f"Expected PIL Image, got {type(img)}")
        img = img.convert("RGB").resize((new_w, new_h), Image.BICUBIC)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_arr = mx.array(img_np).transpose(2, 0, 1)  # (C, H, W)
        img_arr = _normalize_image(img_arr)
        frame_arrays.append(img_arr)

    # Stack: (n_frames, C, H, W)
    all_frames = mx.stack(frame_arrays, axis=0)

    # Group into temporal patches: (n_temporal_groups, temporal_patch_size, C, H, W)
    grouped = all_frames.reshape(n_temporal_groups, temporal_patch_size, 3, new_h, new_w)

    # Patchify each temporal group
    all_patches = []
    for t_idx in range(n_temporal_groups):
        group = grouped[t_idx]  # (temporal_patch_size, C, H, W)
        # -> (C, temporal_patch_size, H, W)
        group = group.transpose(1, 0, 2, 3)
        # -> patches: (grid_h * grid_w, C * temporal_patch_size, patch_H, patch_W)
        patches = group.reshape(3, temporal_patch_size, grid_h, patch_size, grid_w, patch_size)
        patches = patches.transpose(2, 4, 1, 0, 3, 5)  # (gh, gw, T, C, pH, pW)
        patches = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size, patch_size, patch_size)
        all_patches.append(patches)

    pixel_values = mx.concatenate(all_patches, axis=0)
    grid_thw = mx.array([[n_temporal_groups, grid_h, grid_w]], dtype=mx.int32)

    return pixel_values, grid_thw


def preprocess_video_standard(
    frames: list[Any],  # list[PIL.Image.Image]
    image_size: int = 384,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> mx.array:
    """Preprocess video frames using standard image pipeline.

    Each frame is treated as an independent image (resize + crop + normalize).
    Suitable for models that process video frame-by-frame through their
    existing image encoder.

    Returns:
        ``(N_frames, C, H, W)`` tensor.
    """
    from mlxs.models.vision.image_processing import preprocess_standard

    return preprocess_standard(frames, image_size=image_size, mean=mean, std=std)
