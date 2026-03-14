"""Mistral3: dual-mode wrapper (text-only / multimodal) — ModelProtocol + MultimodalModelProtocol.

In TEXT mode (default): strips vision weights, delegates to Ministral3 or Llama text tower.
In MULTIMODAL mode: loads Pixtral ViT encoder + MLP projector and supports
``prepare_inputs`` for image processing (FR12, §7.4, AC11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Mistral3 config: model_type + text_config + optional vision_config."""

    model_type: str = "mistral3"
    text_config: dict[str, Any] | None = None
    vision_config: dict[str, Any] | None = None
    image_token_index: int | None = None
    image_token_id: int | None = None
    vision_feature_layer: int = -1

    def __post_init__(self) -> None:
        if self.text_config is None:
            self.text_config = {}
        if "tie_word_embeddings" not in self.text_config:
            self.text_config["tie_word_embeddings"] = False
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class Model(nn.Module):
    """Mistral3 LM wrapper — dual-mode: text-only or multimodal (§7.4)."""

    def __init__(
        self, args: ModelArgs, *, model_mode: ModelMode = ModelMode.TEXT,
    ) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self._mode = model_mode
        text_config = args.text_config or {}
        inner_type = text_config.get("model_type", "ministral3")
        if inner_type == "ministral3":
            from mlxs.models import ministral3
            inner_args = ministral3.ModelArgs.from_dict(text_config)
            self.language_model = ministral3.Model(inner_args)
        else:
            from mlxs.models import llama
            inner_args = llama.ModelArgs.from_dict(text_config)
            self.language_model = llama.Model(inner_args)

        if model_mode != ModelMode.TEXT and args.vision_config:
            from mlxs.models.vision.pixtral_encoder import PixtralVisionConfig, PixtralVisionModel
            from mlxs.models.vision.projectors import MLPProjector

            vc_dict = args.vision_config
            vc = PixtralVisionConfig(
                **{k: v for k, v in vc_dict.items()
                   if k in PixtralVisionConfig.__dataclass_fields__}
            )
            self.vision_tower = PixtralVisionModel(vc)

            text_hidden = text_config.get("hidden_size", 5120)
            self.multi_modal_projector = MLPProjector(
                in_dim=vc.hidden_size,
                hidden_dim=text_hidden,
                out_dim=text_hidden,
            )
            self._vision_feature_layer = args.vision_feature_layer

        self._image_token_id = args.image_token_index or 10

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
        image_sizes: list[tuple[int, int]] | mx.array | None = None,
        **_kwargs: Any,
    ) -> tuple[mx.array, mx.array | None]:
        """Encode images and merge with text embeddings (§7.4)."""
        if pixel_values is None or self._mode == ModelMode.TEXT:
            return input_ids, None

        from mlxs.models.vision import merge_embeddings

        _, hidden_states = self.vision_tower(
            pixel_values, image_sizes=image_sizes, output_hidden_states=True,
        )
        image_features = hidden_states[self._vision_feature_layer]
        image_features = self.multi_modal_projector(image_features)

        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)

        text_embeds = self.language_model.model.embed_tokens(input_ids)

        merged = merge_embeddings(
            text_embeds, image_features, input_ids, self._image_token_id,
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
        _VISION_PREFIXES = (
            "vision_tower.", "vision_encoder.", "multi_modal_projector.",
            "model.vision_encoder.", "model.vision_projection.",
        )

        if self._mode == ModelMode.TEXT:
            prefix = "language_model."
            inner = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            other = {
                k: v for k, v in weights.items()
                if not k.startswith(prefix)
                and not any(k.startswith(p) for p in _VISION_PREFIXES)
            }
            sanitized = self.language_model.sanitize(inner)
            return {prefix + k: v for k, v in sanitized.items()} | other

        # MULTIMODAL
        lm_weights: dict[str, Any] = {}
        vision_weights: dict[str, Any] = {}
        projector_weights: dict[str, Any] = {}
        prefix = "language_model."

        for k, v in weights.items():
            if k.startswith("model.vision_encoder."):
                vision_weights[k.replace("model.vision_encoder.", "vision_tower.")] = v
            elif k.startswith("vision_tower.vision_model."):
                vision_weights[k.replace("vision_tower.vision_model.", "vision_tower.")] = v
            elif k.startswith("vision_tower."):
                vision_weights[k] = v
            elif k.startswith("model.vision_projection."):
                projector_weights[k.replace("model.vision_projection.", "multi_modal_projector.")] = v
            elif k.startswith("multi_modal_projector."):
                projector_weights[k] = v
            elif k.startswith(prefix):
                lm_weights[k[len(prefix):]] = v
            else:
                lm_weights[k] = v

        sanitized_lm = self.language_model.sanitize(lm_weights)

        if hasattr(self, "vision_tower"):
            vision_weights = self.vision_tower.sanitize(vision_weights)

        result = {prefix + k: v for k, v in sanitized_lm.items()}
        result.update(vision_weights)
        result.update(projector_weights)
        return result
