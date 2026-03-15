"""Tests for shared SigLIP builder helpers introduced in P1."""

from __future__ import annotations

import mlx.core as mx

from mlxs.models.vision.siglip import SigLIPVisionModel
from mlxs.models.vision.siglip_builder import (
    build_siglip_vision_tower,
    normalize_siglip_vision_config,
)


def test_normalize_siglip_vision_config_supports_qwen35_aliases() -> None:
    config = normalize_siglip_vision_config(
        {
            "depth": 2,
            "hidden_size": 64,
            "out_hidden_size": 32,
            "num_heads": 4,
            "image_size": 28,
            "patch_size": 14,
            "in_chans": 3,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "mlp_ratio": 2.0,
        }
    )

    assert config.embed_dim == 64
    assert config.hidden_size == 32
    assert config.in_channels == 3


def test_build_siglip_vision_tower_returns_model() -> None:
    model = build_siglip_vision_tower(
        {
            "depth": 2,
            "embed_dim": 64,
            "hidden_size": 32,
            "num_heads": 4,
            "image_size": 28,
            "patch_size": 14,
            "in_channels": 3,
            "mlp_ratio": 2.0,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }
    )

    assert isinstance(model, SigLIPVisionModel)
    mx.eval(model.parameters())

    pixel_values = mx.zeros((4, 6, 14, 14))
    grid_thw = mx.array([[1, 2, 2]])
    output = model(pixel_values, grid_thw)
    assert output.shape == (1, 32)
