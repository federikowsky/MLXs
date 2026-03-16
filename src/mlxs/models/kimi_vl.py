"""Kimi VL: dual-mode wrapper (text-only / multimodal) — ModelProtocol + MultimodalModelProtocol.

In TEXT mode (default): strips vision weights, wraps DeepSeek V3 text backbone.
In MULTIMODAL mode: loads MoonViT vision encoder and supports ``prepare_inputs``
for image processing (FR12, §7.4, AC11).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.models.base import BaseModelArgs
from mlxs.models.deepseek_v3 import Model as DeepseekV3Model
from mlxs.models.deepseek_v3 import ModelArgs as TextConfig

if TYPE_CHECKING:
    class ModuleBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        def trainable_parameters(self) -> dict[str, Any]: ...
else:
    ModuleBase = nn.Module


def _normalize_kimi_vision_config(vision_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(vision_config)

    alias_map = {
        "num_hidden_layers": "depth",
        "num_attention_heads": "num_heads",
        "in_channels": "num_channels",
    }
    for source_key, target_key in alias_map.items():
        if target_key not in normalized and source_key in normalized:
            normalized[target_key] = normalized[source_key]

    if "embed_dim" not in normalized and "hidden_size" in normalized:
        normalized["embed_dim"] = normalized["hidden_size"]

    return normalized


def _normalize_moon_vit_grid(grid: mx.array | None) -> mx.array:
    if grid is None:
        return mx.array([[1, 1]], dtype=mx.int32)

    if grid.ndim != 2 or grid.shape[1] < 2:
        raise ValueError("Kimi VL requires image_grid_thw/image_grid_hw with at least 2 columns")

    if grid.shape[1] == 2:
        return grid.astype(mx.int32)

    return grid[:, -2:].astype(mx.int32)


@dataclass
class ModelArgs(BaseModelArgs):
    """Kimi VL config: model_type + text_config + optional vision_config."""

    text_config: TextConfig | dict[str, Any] | None = None
    model_type: str = "kimi_vl"
    vision_config: dict[str, Any] | None = None
    image_token_id: int = 151655

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        normalized = dict(params)
        if "image_token_id" not in normalized and "media_placeholder_token_id" in normalized:
            normalized["image_token_id"] = normalized["media_placeholder_token_id"]
        if isinstance(normalized.get("vision_config"), dict):
            normalized["vision_config"] = _normalize_kimi_vision_config(
                normalized["vision_config"]
            )
        allowed = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in normalized.items() if k in allowed})

    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = cast(TextConfig, TextConfig.from_dict(self.text_config))
        if self.text_config is None:
            raise ValueError("kimi_vl requires text_config")


class Model(ModuleBase):
    """Kimi VL LM wrapper — dual-mode: text-only or multimodal (§7.4)."""

    def __init__(
        self, config: ModelArgs, *, model_mode: ModelMode = ModelMode.TEXT,
    ) -> None:
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self._mode = model_mode
        text_config = config.text_config
        if not isinstance(text_config, TextConfig):
            raise TypeError("kimi_vl requires text_config to resolve to DeepseekV3 ModelArgs")
        self.language_model = DeepseekV3Model(text_config)

        if model_mode != ModelMode.TEXT and config.vision_config:
            from mlxs.models.vision.moon_vit import MoonViTConfig, MoonViTModel
            from mlxs.models.vision.projectors import MLPProjector

            vc_dict = _normalize_kimi_vision_config(config.vision_config)
            vc = MoonViTConfig(
                **{k: v for k, v in vc_dict.items()
                   if k in MoonViTConfig.__dataclass_fields__}
            )
            self.vision_tower = MoonViTModel(vc)

            # Projector: vision output → LM hidden
            text_hidden = text_config.hidden_size
            vision_hidden = vc.embed_dim * (vc.spatial_merge_size ** 2)
            self.multi_modal_projector = MLPProjector(
                in_dim=vision_hidden,
                hidden_dim=text_hidden,
                out_dim=text_hidden,
            )

        self._image_token_id = config.image_token_id

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings,
        )

    def prepare_inputs(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        **_kwargs: Any,
    ) -> tuple[mx.array, mx.array | None]:
        """Encode images and merge with text embeddings (§7.4)."""
        if pixel_values is None or self._mode == ModelMode.TEXT:
            return input_ids, None

        from mlxs.models.vision import merge_embeddings

        # MoonViT consumes per-patch NHWC inputs and 2D patch-grid shapes.
        grid = _normalize_moon_vit_grid(image_grid_thw)
        image_embeds_list = self.vision_tower(pixel_values, grid)

        # Concatenate and project
        image_embeds = mx.concatenate(
            [e.reshape(-1, e.shape[-1]) for e in image_embeds_list], axis=0,
        )
        image_embeds = self.multi_modal_projector(image_embeds)

        text_embeds = self.language_model.model.embed_tokens(input_ids)

        merged = merge_embeddings(
            text_embeds, image_embeds, input_ids, self._image_token_id,
        )
        return input_ids, merged

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
        skip_substrings = (
            "vision_tower", "vision_model", "multi_modal_projector",
            "mm_projector", "rotary_emb",
        )

        if self._mode == ModelMode.TEXT:
            filtered = {
                k: v for k, v in weights.items()
                if not any(s in k for s in skip_substrings)
            }
            lm_prefix = "language_model."
            inner = {
                k[len(lm_prefix):]: v
                for k, v in filtered.items() if k.startswith(lm_prefix)
            }
            if not inner and any(k.startswith("model.") for k in filtered):
                inner = dict(filtered)
            if not inner:
                return filtered
            inner = self.language_model.sanitize(inner)
            return {lm_prefix + k: v for k, v in inner.items()}

        # MULTIMODAL
        lm_weights: dict[str, Any] = {}
        vision_weights: dict[str, Any] = {}
        projector_weights: dict[str, Any] = {}
        lm_prefix = "language_model."

        for k, v in weights.items():
            if "rotary_emb" in k:
                continue
            if k.startswith("vision_tower."):
                vision_weights[k] = v
            elif k.startswith("multi_modal_projector.") or k.startswith("mm_projector."):
                projector_weights[k] = v
            elif k.startswith(lm_prefix):
                lm_weights[k[len(lm_prefix):]] = v
            else:
                lm_weights[k] = v

        sanitized_lm = self.language_model.sanitize(lm_weights)

        if hasattr(self, "vision_tower"):
            vision_weights = self.vision_tower.sanitize(vision_weights)

        result = {lm_prefix + k: v for k, v in sanitized_lm.items()}
        result.update(vision_weights)
        result.update(projector_weights)
        return result

    def parameters(self) -> dict[str, Any]:
        return dict(self.trainable_parameters())
