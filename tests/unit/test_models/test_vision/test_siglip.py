"""Tests for SigLIP vision encoder forward pass (§7.4, FR12)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.models.vision.siglip import SigLIPVisionModel, VisionConfig


def _make_tiny_config() -> VisionConfig:
    """Minimal SigLIP config for testing."""
    return VisionConfig(
        depth=2,
        embed_dim=64,
        hidden_size=32,
        num_heads=4,
        image_size=28,
        patch_size=14,
        in_channels=3,
        mlp_ratio=2.0,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )


def test_siglip_forward_shape() -> None:
    """SigLIPVisionModel produces correct output shape."""
    config = _make_tiny_config()
    model = SigLIPVisionModel(config)
    mx.eval(model.parameters())

    # 1 image, grid: temporal=1, height=2, width=2 (2x2 patches)
    # Each patch: C * temporal_patch_size = 3 * 2 = 6 channels, patch_size=14
    n_patches = 1 * 2 * 2  # t * h * w = 4
    pixel_values = mx.zeros((n_patches, 3 * 2, 14, 14))
    grid_thw = mx.array([[1, 2, 2]])

    output = model(pixel_values, grid_thw)
    # After merge: 4 patches / (2*2) = 1 merged patch, hidden_size=32
    assert output.shape == (1, 32)


def test_siglip_multiple_images() -> None:
    """SigLIPVisionModel handles multiple images."""
    config = _make_tiny_config()
    model = SigLIPVisionModel(config)
    mx.eval(model.parameters())

    # 2 images, each 2x2 patches
    n_patches = 2 * (1 * 2 * 2)  # 8 total patches
    pixel_values = mx.zeros((n_patches, 6, 14, 14))
    grid_thw = mx.array([[1, 2, 2], [1, 2, 2]])

    output = model(pixel_values, grid_thw)
    # 8 patches / 4 (merge) = 2 merged patches
    assert output.shape == (2, 32)
