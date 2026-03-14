"""Kimi VL: dual-mode wrapper (text-only / multimodal) — ModelProtocol + MultimodalModelProtocol.

In TEXT mode (default): strips vision weights, wraps DeepSeek V3 text backbone.
In MULTIMODAL mode: loads MoonViT vision encoder and supports ``prepare_inputs``
for image processing (FR12, §7.4, AC11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.models.base import BaseModelArgs
from mlxs.models.deepseek_v3 import Model as DeepseekV3Model
from mlxs.models.deepseek_v3 import ModelArgs as TextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    """Kimi VL config: model_type + text_config + optional vision_config."""

    text_config: TextConfig | dict[str, Any] = None  # type: ignore[assignment]
    model_type: str = "kimi_vl"
    vision_config: dict[str, Any] | None = None
    image_token_id: int = 151655

    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)


class Model(nn.Module):
    """Kimi VL LM wrapper — dual-mode: text-only or multimodal (§7.4)."""

    def __init__(
        self, config: ModelArgs, *, model_mode: ModelMode = ModelMode.TEXT,
    ) -> None:
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self._mode = model_mode
        self.language_model = DeepseekV3Model(config.text_config)

        if model_mode != ModelMode.TEXT and config.vision_config:
            from mlxs.models.vision.moon_vit import MoonViTConfig, MoonViTModel
            from mlxs.models.vision.projectors import MLPProjector

            vc_dict = config.vision_config
            vc = MoonViTConfig(
                **{k: v for k, v in vc_dict.items()
                   if k in MoonViTConfig.__dataclass_fields__}
            )
            self.vision_tower = MoonViTModel(vc)

            # Projector: vision output → LM hidden
            text_hidden = config.text_config.hidden_size
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

        # Vision tower returns list of merged patch tensors
        grid = image_grid_thw if image_grid_thw is not None else mx.array([[1, 1, 1]])
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
        return self.language_model.parameters()
