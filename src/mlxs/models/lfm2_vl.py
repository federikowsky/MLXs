"""LFM2-VL: dual-mode wrapper (text-only / multimodal) — ModelProtocol + MultimodalModelProtocol.

In TEXT mode (default): strips vision weights, wraps LFM2 language model.
In MULTIMODAL mode: loads SigLIP2-style ViT encoder + projector and supports
``prepare_inputs`` for image processing (FR12, §7.4, AC11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.models.base import BaseModelArgs
from mlxs.models.lfm2 import Model as LFM2Model, ModelArgs as LFM2ModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """LFM2-VL config: model_type + text_config + optional vision_config."""

    model_type: str = "lfm2_vl"
    text_config: dict[str, Any] | None = None
    vision_config: dict[str, Any] | None = None
    image_token_id: int = 151655
    vision_feature_layer: int = -1

    def __post_init__(self) -> None:
        if self.text_config is not None:
            self.text_config = {**self.text_config, "tie_word_embeddings": False}


class Model(nn.Module):
    """LFM2-VL LM wrapper — dual-mode: text-only or multimodal (§7.4)."""

    def __init__(
        self, args: ModelArgs, *, model_mode: ModelMode = ModelMode.TEXT,
    ) -> None:
        super().__init__()
        if args.text_config is None:
            raise ValueError("lfm2_vl requires text_config")
        self.args = args
        self.model_type = args.model_type
        self._mode = model_mode
        self.language_model = LFM2Model(LFM2ModelArgs.from_dict(args.text_config))

        if model_mode != ModelMode.TEXT and args.vision_config:
            from mlxs.models.vision.lfm2_vit import LFM2VisionConfig, LFM2VisionModel
            from mlxs.models.vision.projectors import MLPProjector

            vc_dict = args.vision_config
            vc = LFM2VisionConfig(
                **{k: v for k, v in vc_dict.items()
                   if k in LFM2VisionConfig.__dataclass_fields__}
            )
            self.vision_tower = LFM2VisionModel(vc)

            text_hidden = args.text_config.get("hidden_size", 2048)
            self.multi_modal_projector = MLPProjector(
                in_dim=vc.hidden_size,
                hidden_dim=text_hidden,
                out_dim=text_hidden,
            )
            self._vision_feature_layer = args.vision_feature_layer

        self._image_token_id = args.image_token_id

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
        **_kwargs: Any,
    ) -> tuple[mx.array, mx.array | None]:
        """Encode images and merge with text embeddings (§7.4)."""
        if pixel_values is None or self._mode == ModelMode.TEXT:
            return input_ids, None

        from mlxs.models.vision import merge_embeddings

        encoder_outputs, _, last_hidden = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(last_hidden)

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
        drop = ("vision_tower", "multi_modal_projector")

        if self._mode == ModelMode.TEXT:
            return {
                k: v for k, v in weights.items()
                if k not in drop and not any(k.startswith(p + ".") for p in drop)
            }

        # MULTIMODAL
        lm_weights: dict[str, Any] = {}
        vision_weights: dict[str, Any] = {}
        projector_weights: dict[str, Any] = {}

        for k, v in weights.items():
            if k.startswith("vision_tower."):
                vision_weights[k] = v
            elif k.startswith("multi_modal_projector."):
                projector_weights[k] = v
            else:
                lm_weights[k] = v

        if hasattr(self, "vision_tower"):
            vision_weights = self.vision_tower.sanitize(vision_weights)

        result = dict(lm_weights)
        result.update(vision_weights)
        result.update(projector_weights)
        return result

    def parameters(self) -> dict[str, Any]:
        return dict(self.trainable_parameters())
