"""Mistral3 wrapper — port from mlx_lm (mlx_lm/models/mistral3.py), ModelProtocol-compliant.

Mistral3 is a container that delegates to a text tower: either Ministral3 or Llama
based on text_config.model_type. Used for multimodal checkpoints; text-only configs
use text_config with model_type \"ministral3\" or \"llama\". Imports: mlxs.cache (via
inner model), mlxs.layers (via inner model), mlxs.models.base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Mistral3 config: model_type + text_config (nested text tower config)."""

    model_type: str = "mistral3"
    text_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.text_config is None:
            self.text_config = {}
        if "tie_word_embeddings" not in self.text_config:
            self.text_config["tie_word_embeddings"] = False


class Model(nn.Module):
    """Mistral3 LM wrapper — delegates to Ministral3 or Llama per text_config.model_type."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
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

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[Any] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        return self.language_model(input_ids, cache=cache, mask=mask)

    @property
    def num_layers(self) -> int:
        return self.language_model.num_layers

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    def make_cache(self) -> list[Any]:
        return self.language_model.make_cache()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        prefix = "language_model."
        inner = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        drop_prefixes = ("vision_tower.", "multi_modal_projector.")
        other = {
            k: v
            for k, v in weights.items()
            if not k.startswith(prefix) and not any(k.startswith(p) for p in drop_prefixes)
        }
        sanitized_inner = self.language_model.sanitize(inner)
        return {prefix + k: v for k, v in sanitized_inner.items()} | other
