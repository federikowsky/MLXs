"""Qwen3 VL MoE: dual-mode wrapper (text-only / multimodal) — ModelProtocol + MultimodalModelProtocol.

In TEXT mode (default): strips vision weights, wraps Qwen3 MoE text backbone.
In MULTIMODAL mode: loads SigLIP vision encoder and supports ``prepare_inputs``
for image processing (FR12, §7.4, AC11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._types import ModelMode
from mlxs.cache.kv import KVCache
from mlxs.models import qwen3_moe
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen3 VL MoE config: model_type + text_config + optional vision_config."""

    model_type: str = "qwen3_vl_moe"
    text_config: dict[str, Any] | None = None
    vision_config: dict[str, Any] | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        if "text_config" in params:
            return cls(
                model_type=params.get("model_type", "qwen3_vl_moe"),
                text_config=params["text_config"],
                vision_config=params.get("vision_config"),
                image_token_id=params.get("image_token_id", 151655),
                video_token_id=params.get("video_token_id", 151656),
            )
        excluded = {"model_type", "vision_config", "image_token_id", "video_token_id"}
        return cls(
            model_type=params.get("model_type", "qwen3_vl_moe"),
            text_config={k: v for k, v in params.items() if k not in excluded},
            vision_config=params.get("vision_config"),
            image_token_id=params.get("image_token_id", 151655),
            video_token_id=params.get("video_token_id", 151656),
        )


class Model(nn.Module):
    """Qwen3 VL MoE LM wrapper — dual-mode: text-only or multimodal (§7.4)."""

    def __init__(
        self, args: ModelArgs, *, model_mode: ModelMode = ModelMode.TEXT,
    ) -> None:
        super().__init__()
        if args.text_config is None:
            raise ValueError("qwen3_vl_moe requires text_config")
        self.args = args
        self.model_type = args.model_type
        self._mode = model_mode
        text_args = qwen3_moe.ModelArgs.from_dict(args.text_config)
        self.language_model = qwen3_moe.Model(text_args)

        if model_mode != ModelMode.TEXT and args.vision_config:
            from mlxs.models.vision.siglip import SigLIPVisionModel, VisionConfig

            vc_dict = args.vision_config
            vc = VisionConfig(
                **{k: v for k, v in vc_dict.items() if k in VisionConfig.__dataclass_fields__}
            )
            self.vision_tower = SigLIPVisionModel(vc)

        self._image_token_id = args.image_token_id

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings, **_kwargs,
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

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        grid = image_grid_thw if image_grid_thw is not None else mx.array([[1, 1, 1]])
        image_embeds = self.vision_tower(pixel_values, grid)

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

    def make_cache(self) -> list[KVCache]:
        return self.language_model.make_cache()

    @property
    def layers(self) -> Any:
        return self.language_model.layers

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        _VISION_PREFIXES = (
            "visual.", "vision_tower.", "vision_model.",
            "multi_modal_projector.", "mm_projector.",
        )
        _VISION_EXACT = ("visual", "vision_tower", "vision_model")

        if self._mode == ModelMode.TEXT:
            filtered = {
                k: v for k, v in weights.items()
                if k not in _VISION_EXACT
                and not any(k.startswith(p) for p in _VISION_PREFIXES)
            }
        else:
            filtered = dict(weights)

        lm_prefix = "language_model."

        if self._mode == ModelMode.TEXT:
            # Normalize keys: ensure language_model.* prefix
            normalized: dict[str, Any] = {}
            for k, v in filtered.items():
                if not k.startswith(lm_prefix):
                    normalized[lm_prefix + k] = v
                else:
                    normalized[k] = v

            # Expert stacking
            num_hidden_layers = self.language_model.args.num_hidden_layers
            num_experts = self.language_model.args.num_experts
            prefix = f"{lm_prefix}model.layers"
            for layer_idx in range(num_hidden_layers):
                layer_prefix = f"{prefix}.{layer_idx}.mlp"
                for n in ["up_proj", "down_proj", "gate_proj"]:
                    key_0 = f"{layer_prefix}.experts.0.{n}.weight"
                    if key_0 not in normalized:
                        continue
                    to_join = [
                        normalized.pop(f"{layer_prefix}.experts.{e}.{n}.weight")
                        for e in range(num_experts)
                    ]
                    normalized[f"{layer_prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

            if self.language_model.args.tie_word_embeddings:
                normalized.pop(f"{lm_prefix}lm_head.weight", None)
            return normalized

        # MULTIMODAL: separate LM and vision weights
        lm_weights: dict[str, Any] = {}
        vision_weights: dict[str, Any] = {}
        for k, v in filtered.items():
            if k.startswith("visual."):
                k = "vision_tower." + k[len("visual."):]

            if k.startswith("vision_tower."):
                vision_weights[k] = v
            elif k.startswith(lm_prefix):
                lm_weights[k[len(lm_prefix):]] = v
            elif k.startswith("model.") or k.startswith("lm_head."):
                lm_weights[k] = v
            else:
                lm_weights[k] = v

        # Expert stacking on LM weights
        num_hidden_layers = self.language_model.args.num_hidden_layers
        num_experts = self.language_model.args.num_experts
        for layer_idx in range(num_hidden_layers):
            layer_prefix = f"model.layers.{layer_idx}.mlp"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                key_0 = f"{layer_prefix}.experts.0.{n}.weight"
                if key_0 not in lm_weights:
                    continue
                to_join = [
                    lm_weights.pop(f"{layer_prefix}.experts.{e}.{n}.weight")
                    for e in range(num_experts)
                ]
                lm_weights[f"{layer_prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        if self.language_model.args.tie_word_embeddings:
            lm_weights.pop("lm_head.weight", None)

        if hasattr(self, "vision_tower"):
            vision_weights = self.vision_tower.sanitize(vision_weights)

        result = {lm_prefix + k: v for k, v in lm_weights.items()}
        result.update(vision_weights)
        return result
