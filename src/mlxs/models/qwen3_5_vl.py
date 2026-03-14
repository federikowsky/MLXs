"""Qwen3.5-VL dual-mode wrapper — ModelProtocol + MultimodalModelProtocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.models.base import BaseModelArgs
from mlxs.models.qwen3_5 import Model as Qwen35Model
from mlxs.models.qwen3_5 import ModelArgs as Qwen35ModelArgs

# Sources and design notes:
# - HF `Qwen/Qwen3.5-0.8B` config.json uses nested `text_config` +
#   `vision_config`, `image_token_id=248056`, `video_token_id=248057`, and
#   multimodal weights split between `language_model.*` and `visual.*`.
# - HF `Qwen/Qwen2.5-VL-7B-Instruct` and `Qwen/Qwen3-VL-8B-Instruct` configs
#   use the same placeholder-token layout and ViT-style vision settings.
# - Local mlx_lm references:
#   `.venv/lib/python3.13/site-packages/mlx_lm/models/qwen3_5.py`,
#   `.venv/lib/python3.13/site-packages/mlx_lm/models/qwen2_vl.py`, and
#   `.venv/lib/python3.13/site-packages/mlx_lm/models/qwen3_vl.py`.
# Inference from those sources: Qwen3.5 multimodal checkpoints follow the same
# text/vision split as earlier Qwen VL models, so this wrapper reuses the
# repo's SigLIP-style vision tower and normalizes HF `vision_config` keys
# (`hidden_size` -> encoder width, `out_hidden_size` -> LM hidden size).

_VISION_PREFIXES = (
    "visual.",
    "vision_tower.",
    "vision_model.",
    "multi_modal_projector.",
    "mm_projector.",
)
_VISION_EXACT = ("visual", "vision_tower", "vision_model")


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen3.5-VL config: model_type + text_config + optional vision_config."""

    model_type: str = "qwen3_5_vl"
    text_config: dict[str, Any] | None = None
    vision_config: dict[str, Any] | None = None
    image_token_id: int = 248056
    video_token_id: int = 248057

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        vision_config = params.get("vision_config", params.get("visual_config"))
        if "text_config" in params:
            return cls(
                model_type=params.get("model_type", "qwen3_5_vl"),
                text_config=params["text_config"],
                vision_config=vision_config,
                image_token_id=params.get("image_token_id", 248056),
                video_token_id=params.get("video_token_id", 248057),
            )

        excluded = {
            "model_type",
            "vision_config",
            "visual_config",
            "image_token_id",
            "video_token_id",
        }
        text_config = {k: v for k, v in params.items() if k not in excluded}
        return cls(
            model_type=params.get("model_type", "qwen3_5_vl"),
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=params.get("image_token_id", 248056),
            video_token_id=params.get("video_token_id", 248057),
        )


class Model(nn.Module):
    """Qwen3.5-VL wrapper with text-only and multimodal modes."""

    def __init__(
        self,
        args: ModelArgs,
        *,
        model_mode: ModelMode = ModelMode.TEXT,
    ) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self._mode = model_mode
        self._image_token_id = args.image_token_id
        self._video_token_id = args.video_token_id

        text_args = Qwen35ModelArgs.from_dict(args.text_config or {})
        self.language_model = Qwen35Model(text_args)

        if model_mode != ModelMode.TEXT and args.vision_config:
            from mlxs.models.vision.siglip import SigLIPVisionModel

            self.vision_tower = SigLIPVisionModel(_build_vision_config(args.vision_config))

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
        )

    def prepare_inputs(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        video_pixel_values: mx.array | None = None,
        **_kwargs: Any,
    ) -> tuple[mx.array, mx.array | None]:
        if not self.supports_vision or (pixel_values is None and video_pixel_values is None):
            return input_ids, None

        from mlxs.models.vision import merge_embeddings

        text_embeddings = self.language_model.model.embed_tokens(input_ids)
        merged_embeddings = text_embeddings

        if pixel_values is not None:
            image_embeddings = self._encode_media(
                pixel_values,
                image_grid_thw,
            )
            merged_embeddings = merge_embeddings(
                merged_embeddings,
                image_embeddings,
                input_ids,
                self._image_token_id,
            )

        if video_pixel_values is not None:
            video_grid_thw = _kwargs.get("video_grid_thw", image_grid_thw)
            video_embeddings = self._encode_media(
                video_pixel_values,
                video_grid_thw,
            )
            merged_embeddings = merge_embeddings(
                merged_embeddings,
                video_embeddings,
                input_ids,
                self._video_token_id,
            )

        return input_ids, merged_embeddings

    def _encode_media(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array | None,
    ) -> mx.array:
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        media = pixel_values.astype(dtype)
        grid = grid_thw if grid_thw is not None else mx.array([[1, 1, 1]])
        return self.vision_tower(media, grid)

    @property
    def supports_vision(self) -> bool:
        return self._mode != ModelMode.TEXT and hasattr(self, "vision_tower")

    @property
    def supports_audio(self) -> bool:
        return False

    @property
    def image_token_id(self) -> int | None:
        return self._image_token_id if self.supports_vision else None

    @property
    def num_layers(self) -> int:
        return self.language_model.num_layers

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    def make_cache(self) -> list[Any]:
        return self.language_model.make_cache()

    @property
    def layers(self) -> Any:
        return self.language_model.layers

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if self._mode == ModelMode.TEXT:
            filtered = {
                key: value
                for key, value in weights.items()
                if key not in _VISION_EXACT
                and not any(key.startswith(prefix) for prefix in _VISION_PREFIXES)
            }
            inner = {
                (
                    key[len("language_model.") :] if key.startswith("language_model.") else key
                ): value
                for key, value in filtered.items()
            }
            sanitized = self.language_model.sanitize(inner)
            return {f"language_model.{key}": value for key, value in sanitized.items()}

        language_weights: dict[str, Any] = {}
        vision_weights: dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith(("multi_modal_projector.", "mm_projector.")):
                continue
            if key.startswith("visual."):
                key = f"vision_tower.{key[len('visual.') :]}"
            elif key.startswith("vision_model."):
                key = f"vision_tower.{key[len('vision_model.') :]}"

            if key.startswith("vision_tower."):
                vision_weights[key] = value
            elif key.startswith("language_model."):
                language_weights[key[len("language_model.") :]] = value
            else:
                language_weights[key] = value

        sanitized_language = self.language_model.sanitize(language_weights)
        if hasattr(self, "vision_tower"):
            vision_weights = self.vision_tower.sanitize(vision_weights)

        result = {f"language_model.{key}": value for key, value in sanitized_language.items()}
        result.update(vision_weights)
        return result


def _build_vision_config(raw_config: dict[str, Any]) -> Any:
    from mlxs.models.vision.siglip import VisionConfig

    mapped = dict(raw_config)
    mapped["embed_dim"] = raw_config.get("embed_dim", raw_config.get("hidden_size"))
    mapped["hidden_size"] = raw_config.get(
        "out_hidden_size",
        raw_config.get("hidden_size", VisionConfig.hidden_size),
    )
    mapped["in_channels"] = raw_config.get(
        "in_channels",
        raw_config.get("in_chans", VisionConfig.in_channels),
    )
    filtered = {
        key: value for key, value in mapped.items() if key in VisionConfig.__dataclass_fields__
    }
    return VisionConfig(**filtered)
