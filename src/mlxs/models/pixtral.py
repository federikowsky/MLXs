"""Pixtral vision-language model — text backbone only (Llama); text-only path supported.

Implements ModelProtocol. Wraps Llama with text_config; VL path can inject
input_embeddings later. Compatible with mlx_lm-converted weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlxs.cache.kv import KVCache
from mlxs.models.base import BaseModelArgs
from mlxs.models.llama import Model as LlamaModel
from mlxs.models.llama import ModelArgs as LlamaModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Pixtral model configuration; text backbone uses text_config (Llama)."""

    model_type: str = "pixtral"
    text_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.text_config is None:
            self.text_config = {}
        self.text_config["tie_word_embeddings"] = False
        self.text_config.setdefault("num_attention_heads", 32)


class Model(LlamaModel):
    """Pixtral wrapper around Llama text backbone; satisfies ModelProtocol.

    Text-only: forward with input_ids and cache. VL: forward with
    input_embeddings (and optional input_ids for position) when supported.
    """

    def __init__(self, args: ModelArgs) -> None:
        if args.text_config is None:
            args.text_config = {}
        llama_args = LlamaModelArgs.from_dict(args.text_config)
        super().__init__(llama_args)
        self.model_type = args.model_type

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return super().__call__(
            input_ids,
            cache=cache,
            input_embeddings=input_embeddings,
        )

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        out = {
            k: v
            for k, v in weights.items()
            if not k.startswith("vision_tower.") and not k.startswith("multi_modal_projector.")
        }
        # Pixtral checkpoints use "language_model.*"; we are a Llama so strip prefix.
        prefix = "language_model."
        stripped = {k[len(prefix) :]: v for k, v in out.items() if k.startswith(prefix)}
        if stripped:
            out = stripped
        return super().sanitize(out)
