"""Kimi K2.5: text-only wrapper around DeepSeek V3 backbone.

Implements ModelProtocol. Config via model_type + text_config (same as mlx_lm).
Uses mlxs.models.deepseek_v3 for the language backbone. Sanitize strips
vision-related weight keys for compatibility with multi-modal checkpoints.
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
    """Kimi K2.5 config: wraps DeepSeek V3 text_config."""

    text_config: TextConfig | dict[str, Any]
    model_type: str = "kimi_k25"

    def __post_init__(self) -> None:
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)


class Model(nn.Module):
    """Kimi K2.5 model: DeepSeek V3 language backbone only (ModelProtocol)."""

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
        # Drop vision-related top-level keys (multi-modal checkpoints)
        skip_prefixes = (
            "vision_tower.",
            "vision_model.",
            "multi_modal_projector.",
            "mm_projector.",
        )
        filtered = {
            k: v for k, v in weights.items() if not any(k.startswith(p) for p in skip_prefixes)
        }
        # Restrict to language_model subtree and strip prefix for inner sanitize
        lm_prefix = "language_model."
        inner = {k[len(lm_prefix) :]: v for k, v in filtered.items() if k.startswith(lm_prefix)}
        if not inner:
            return filtered
        inner = self.language_model.sanitize(inner)
        # Return inner keys as-is so loader matches language_model.parameters()
        return inner

    def parameters(self) -> dict[str, Any]:
        return self.language_model.parameters()
