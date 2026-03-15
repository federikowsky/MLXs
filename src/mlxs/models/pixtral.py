"""Pixtral: dual-mode wrapper (text-only / multimodal) — ModelProtocol + MultimodalModelProtocol.

In TEXT mode (default): strips vision weights, wraps Llama text backbone.
In MULTIMODAL mode: loads Pixtral ViT encoder + MLP projector and supports
``prepare_inputs`` for image processing (FR12, §7.4, AC11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.cache.kv import KVCache
from mlxs.models.base import BaseModelArgs
from mlxs.models.llama import Model as LlamaModel
from mlxs.models.llama import ModelArgs as LlamaModelArgs
from mlxs.models.multimodal_shared import (
    MediaBranch,
    build_dual_mode_components,
    prepare_multimodal_inputs,
)

if TYPE_CHECKING:
    from mlxs.models.vision.pixtral_encoder import PixtralVisionModel
    from mlxs.models.vision.projectors import MLPProjector


def _build_vision_components(
    raw_config: dict[str, Any],
    *,
    text_hidden: int,
) -> tuple[PixtralVisionModel, MLPProjector]:
    from mlxs.models.vision.pixtral_encoder import PixtralVisionConfig, PixtralVisionModel
    from mlxs.models.vision.projectors import MLPProjector

    config = PixtralVisionConfig(
        **{
            key: value
            for key, value in raw_config.items()
            if key in PixtralVisionConfig.__dataclass_fields__
        }
    )
    vision_tower = PixtralVisionModel(config)
    projector = MLPProjector(
        in_dim=config.hidden_size,
        hidden_dim=text_hidden,
        out_dim=text_hidden,
    )
    return vision_tower, projector


@dataclass
class ModelArgs(BaseModelArgs):
    """Pixtral config: model_type + text_config + optional vision_config."""

    model_type: str = "pixtral"
    text_config: dict[str, Any] | None = None
    vision_config: dict[str, Any] | None = None
    image_token_index: int | None = None
    image_token_id: int | None = None
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "full"

    def __post_init__(self) -> None:
        if self.text_config is None:
            self.text_config = {}
        self.text_config["tie_word_embeddings"] = False
        self.text_config.setdefault("num_attention_heads", 32)
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id


class Model(nn.Module):
    """Pixtral LM wrapper — dual-mode: text-only or multimodal (§7.4)."""

    def __init__(
        self, args: ModelArgs, *, model_mode: ModelMode = ModelMode.TEXT,
    ) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        llama_args = LlamaModelArgs.from_dict(args.text_config or {})
        components = build_dual_mode_components(
            model_mode=model_mode,
            language_builder=lambda: LlamaModel(llama_args),
            vision_config=args.vision_config,
            vision_builder=lambda raw_config: _build_vision_components(
                raw_config,
                text_hidden=(args.text_config or {}).get("hidden_size", 5120),
            ),
        )
        self._mode = components.model_mode
        self.language_model = components.language_model
        if components.vision_tower is not None:
            self.vision_tower, self.multi_modal_projector = components.vision_tower
            self._vision_feature_layer = args.vision_feature_layer

        self._image_token_id = args.image_token_index or 10

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
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

        image_branch = MediaBranch(
            values=pixel_values,
            placeholder_token_id=self._image_token_id,
            encode=self._encode_image,
            encoder_kwargs={"image_sizes": image_sizes},
        )
        return prepare_multimodal_inputs(
            input_ids,
            embed_tokens=self.language_model.model.embed_tokens,
            image_branch=image_branch,
        )

    def _encode_image(
        self,
        pixel_values: mx.array,
        *,
        image_sizes: list[tuple[int, int]] | mx.array | None = None,
    ) -> mx.array:
        _, hidden_states = self.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
        )
        image_features = hidden_states[self._vision_feature_layer]
        image_features = self.multi_modal_projector(image_features)
        return image_features.reshape(-1, image_features.shape[-1])

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

    def make_cache(self) -> list[KVCache]:
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
            out = {
                k: v for k, v in weights.items()
                if not any(k.startswith(p) for p in _VISION_PREFIXES)
            }
            prefix = "language_model."
            stripped = {k[len(prefix):]: v for k, v in out.items() if k.startswith(prefix)}
            if stripped:
                out = stripped
            return self.language_model.sanitize(out)

        # MULTIMODAL: remap keys
        lm_weights: dict[str, Any] = {}
        vision_weights: dict[str, Any] = {}
        projector_weights: dict[str, Any] = {}

        for k, v in weights.items():
            if k.startswith("model.vision_encoder."):
                vision_weights[k.replace("model.vision_encoder.", "vision_tower.")] = v
            elif k.startswith("vision_tower.vision_model."):
                vision_weights[k.replace("vision_tower.vision_model.", "vision_tower.")] = v
            elif k.startswith("vision_tower."):
                vision_weights[k] = v
            elif k.startswith("model.vision_projection."):
                projector_key = k.replace(
                    "model.vision_projection.",
                    "multi_modal_projector.",
                )
                projector_weights[projector_key] = v
            elif k.startswith("multi_modal_projector."):
                projector_weights[k] = v
            elif k.startswith("language_model."):
                lm_weights[k[len("language_model."):]] = v
            elif k.startswith("model.language_model."):
                lm_weights[k.replace("model.language_model.", "model.")] = v
            else:
                lm_weights[k] = v

        sanitized_lm = self.language_model.sanitize(lm_weights)

        if hasattr(self, "vision_tower"):
            vision_weights = self.vision_tower.sanitize(vision_weights)

        result = {"language_model." + k: v for k, v in sanitized_lm.items()}
        result.update(vision_weights)
        result.update(projector_weights)
        return result
