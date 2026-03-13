"""Kimi VL: text-only wrapper around DeepSeek V3 backbone.

Ported from mlx_lm. Implements ModelProtocol. Vision/MM keys are dropped
in sanitize so text-only loading works. Uses mlxs.models.deepseek_v3 for
the language backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.models.base import BaseModelArgs
from mlxs.models.deepseek_v3 import Model as DeepseekV3Model
from mlxs.models.deepseek_v3 import ModelArgs as TextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    """Kimi VL config: model_type + text_config (DeepSeek V3)."""

    text_config: TextConfig | dict[str, Any]
    model_type: str = "kimi_vl"

    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)


class Model(nn.Module):
    """Kimi VL: DeepSeek V3 language backbone only (ModelProtocol)."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.language_model = DeepseekV3Model(config.text_config)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[Any] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        return self.language_model(input_ids, cache=cache)

    @property
    def num_layers(self) -> int:
        return self.language_model.num_layers

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    def make_cache(self) -> list[Any]:
        return self.language_model.make_cache()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        skip_substrings = (
            "vision_tower",
            "vision_model",
            "multi_modal_projector",
            "mm_projector",
            "rotary_emb",
        )
        filtered = {k: v for k, v in weights.items() if not any(s in k for s in skip_substrings)}
        lm_prefix = "language_model."
        inner = {k[len(lm_prefix) :]: v for k, v in filtered.items() if k.startswith(lm_prefix)}
        if not inner and any(k.startswith("model.") for k in filtered):
            inner = dict(filtered)
        if not inner:
            return filtered
        inner = self.language_model.sanitize(inner)
        return {(lm_prefix + k): v for k, v in inner.items()}

    def parameters(self) -> dict[str, Any]:
        return self.language_model.parameters()
