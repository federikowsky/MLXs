"""Qwen2-VL: text-only wrapper around Qwen2 backbone (ModelProtocol).

Port from mlx_lm. VL checkpoints include vision; we only implement the text path.
Config via model_type + text_config (or flat params as text_config). Compatible
with mlx_lm-converted weights; sanitize strips vision keys and normalizes to
language_model.* for loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.models.base import BaseModelArgs
from mlxs.models.qwen import Model as Qwen2Model
from mlxs.models.qwen import ModelArgs as Qwen2ModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen2-VL config: model_type + text_config (Qwen2 text backbone)."""

    model_type: str = "qwen2_vl"
    text_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        if "text_config" in params:
            return cls(
                model_type=params.get("model_type", "qwen2_vl"),
                text_config=params["text_config"],
            )
        # Flat config: use full params as text_config (mlx_lm compatibility)
        text_config = {k: v for k, v in params.items() if k != "model_type"}
        return cls(
            model_type=params.get("model_type", "qwen2_vl"),
            text_config=text_config,
        )


class Model(nn.Module):
    """Qwen2-VL LM: Qwen2 language backbone only — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_config = args.text_config or {}
        self.language_model = Qwen2Model(Qwen2ModelArgs.from_dict(text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(inputs, cache=cache, **_kwargs)

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
        # Drop vision-related top-level keys (VL checkpoints)
        drop = ("visual", "vision_tower", "vision_model")
        filtered = {k: v for k, v in weights.items() if k not in drop}
        drop_prefixes = (
            "visual.",
            "vision_tower.",
            "vision_model.",
            "multi_modal_projector.",
            "mm_projector.",
        )
        filtered = {
            k: v for k, v in filtered.items() if not any(k.startswith(p) for p in drop_prefixes)
        }
        lm_prefix = "language_model."
        inner = {
            (k[len(lm_prefix) :] if k.startswith(lm_prefix) else k): v for k, v in filtered.items()
        }
        sanitized_inner = self.language_model.sanitize(inner)
        return {lm_prefix + k: v for k, v in sanitized_inner.items()}
