"""Tests for shared multimodal helpers introduced in P1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.models.multimodal_shared import (
    MediaBranch,
    MultimodalArgsMixin,
    build_dual_mode_components,
    prepare_multimodal_inputs,
)


@dataclass
class _Args(MultimodalArgsMixin[dict[str, Any]]):
    model_type: str = "mock_vl"
    vision_feature_layer: int = -1


@dataclass(frozen=True)
class _TextConfig:
    hidden_size: int


def test_multimodal_args_from_nested_config() -> None:
    args = _Args.from_flat_or_nested(
        {
            "model_type": "mock_vl",
            "text_config": {"hidden_size": 64},
            "vision_config": {"depth": 2},
            "image_token_id": 10,
            "video_token_id": 11,
            "vision_feature_layer": -2,
        },
        default_model_type="mock_vl",
    )

    assert args.model_type == "mock_vl"
    assert args.text_config == {"hidden_size": 64}
    assert args.vision_config == {"depth": 2}
    assert args.image_token_id == 10
    assert args.video_token_id == 11
    assert args.vision_feature_layer == -2


def test_multimodal_args_from_flat_config_respects_excluded_and_top_level() -> None:
    args = _Args.from_flat_or_nested(
        {
            "model_type": "mock_vl",
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "vision_config": {"depth": 1},
            "image_token_id": 99,
            "vision_feature_layer": -3,
            "ignored_runtime_flag": True,
        },
        default_model_type="mock_vl",
        excluded_keys={"ignored_runtime_flag"},
    )

    assert args.text_config == {"hidden_size": 128, "num_hidden_layers": 4}
    assert args.vision_config == {"depth": 1}
    assert args.image_token_id == 99
    assert args.vision_feature_layer == -3


def test_multimodal_args_supports_vision_aliases_and_text_normalizer() -> None:
    args = _Args.from_flat_or_nested(
        {
            "hidden_size": 256,
            "visual_config": {"depth": 3},
        },
        default_model_type="mock_vl",
        vision_config_keys=("vision_config", "visual_config"),
        text_config_normalizer=lambda cfg: {"wrapped_hidden_size": cfg["hidden_size"]},
    )

    assert args.text_config == {"wrapped_hidden_size": 256}
    assert args.vision_config == {"depth": 3}


def test_multimodal_args_can_convert_text_config_type() -> None:
    @dataclass
    class _TypedArgs(MultimodalArgsMixin[_TextConfig]):
        model_type: str = "typed_vl"

    args = _TypedArgs.from_flat_or_nested(
        {"text_config": {"hidden_size": 42}},
        default_model_type="typed_vl",
        text_config_normalizer=lambda cfg: _TextConfig(hidden_size=int(cfg["hidden_size"])),
    )

    assert args.text_config == _TextConfig(hidden_size=42)


def test_dual_mode_components_build_language_side_always() -> None:
    calls: list[str] = []

    def build_language() -> str:
        calls.append("language")
        return "lm"

    def build_vision(config: dict[str, Any]) -> str:
        calls.append(f"vision:{config['depth']}")
        return "vit"

    components = build_dual_mode_components(
        model_mode=ModelMode.TEXT,
        language_builder=build_language,
        vision_config={"depth": 2},
        vision_builder=build_vision,
    )

    assert components.language_model == "lm"
    assert components.vision_tower is None
    assert components.supports_vision is False
    assert calls == ["language"]


def test_dual_mode_components_build_vision_only_when_enabled() -> None:
    components = build_dual_mode_components(
        model_mode=ModelMode.MULTIMODAL,
        language_builder=lambda: "lm",
        vision_config={"depth": 2},
        vision_builder=lambda config: f"vit:{config['depth']}",
    )

    assert components.language_model == "lm"
    assert components.vision_tower == "vit:2"
    assert components.supports_vision is True


def test_prepare_multimodal_inputs_returns_none_without_media() -> None:
    input_ids = mx.array([[1, 2, 3]])
    token_embeds = mx.reshape(mx.arange(12, dtype=mx.float32), (1, 3, 4))

    ids_out, merged = prepare_multimodal_inputs(
        input_ids,
        embed_tokens=lambda _ids: token_embeds,
    )

    assert mx.array_equal(ids_out, input_ids)
    assert merged is None


def test_prepare_multimodal_inputs_merges_image_and_video_in_order() -> None:
    input_ids = mx.array([[10, 1, 20, 2]])
    text_embeds = mx.zeros((1, 4, 3))

    image_embeds = mx.array([[1.0, 1.0, 1.0]])
    video_embeds = mx.array([[2.0, 2.0, 2.0]])

    ids_out, merged = prepare_multimodal_inputs(
        input_ids,
        embed_tokens=lambda _ids: text_embeds,
        image_branch=MediaBranch(
            values=mx.array([[0.0]]),
            placeholder_token_id=10,
            encode=lambda _values: image_embeds,
        ),
        video_branch=MediaBranch(
            values=mx.array([[0.0]]),
            placeholder_token_id=20,
            encode=lambda _values, grid=None: video_embeds,
            encoder_kwargs={"grid": mx.array([[1, 1, 1]])},
        ),
    )

    assert mx.array_equal(ids_out, input_ids)
    assert merged is not None
    assert merged.shape == (1, 4, 3)
    assert mx.array_equal(merged[0, 0], image_embeds[0])
    assert mx.array_equal(merged[0, 2], video_embeds[0])

