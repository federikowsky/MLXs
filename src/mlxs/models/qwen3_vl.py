"""Qwen3-VL text backbone — ModelProtocol, text-only path.

Wraps the Qwen3 language model; vision tower and fusion are out of scope.
Checkpoint keys: strip vision_tower, keep language_model.* for loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.models.qwen3 import Model as Qwen3Model
from mlxs.models.qwen3 import ModelArgs as Qwen3ModelArgs

if TYPE_CHECKING:
    from mlxs.protocols.cache import CacheProtocol


@dataclass
class ModelArgs:
    """Qwen3-VL config: model_type + text_config for the language backbone."""

    model_type: str = "qwen3_vl"
    text_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        if "text_config" in params:
            return cls(
                model_type=params.get("model_type", "qwen3_vl"),
                text_config=params["text_config"],
            )
        return cls(
            model_type=params.get("model_type", "qwen3_vl"),
            text_config={k: v for k, v in params.items() if k != "model_type"},
        )


class Model(nn.Module):
    """Qwen3-VL wrapper: text backbone only; satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        text = args.text_config or {}
        self.model_type = args.model_type
        self.language_model = Qwen3Model(Qwen3ModelArgs.from_dict(text))

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[Any] | None = None,
        mask: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(input_ids, cache=cache, **_kwargs)

    @property
    def num_layers(self) -> int:
        return self.language_model.num_layers

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    def make_cache(self) -> list[CacheProtocol]:
        return self.language_model.make_cache()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in weights.items():
            if k.startswith("vision_tower") or ".vision_tower." in k:
                continue
            if k.startswith("language_model."):
                out[k] = v
            else:
                out["language_model." + k] = v
        return out
