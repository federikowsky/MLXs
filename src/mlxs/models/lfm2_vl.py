"""LFM2-VL: text backbone only; implements ModelProtocol.

Port from mlx_lm. VL model wraps LFM2 language model; text-only path uses
the same backbone. Vision tower and multi_modal_projector are stripped in
sanitize and not used. Registry is used to resolve the LFM2 backbone (no
cross-model import).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.load.registry import get_model_classes
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """LFM2-VL config: model_type + text_config (nested LFM2 config)."""

    model_type: str = "lfm2_vl"
    text_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.text_config is not None:
            self.text_config = {**self.text_config, "tie_word_embeddings": False}


class Model(nn.Module):
    """LFM2-VL top-level: delegates to LFM2 text backbone. Implements ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        if args.text_config is None:
            raise ValueError("lfm2_vl requires text_config")
        ModelCls, ArgsCls = get_model_classes("lfm2")
        self.args = args
        self.model_type = args.model_type
        self.language_model = ModelCls(ArgsCls.from_dict(args.text_config))

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[Any] | None = None,
        mask: mx.array | None = None,
        **_kwargs: Any,
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
        drop = ("vision_tower", "multi_modal_projector")
        return {
            k: v
            for k, v in weights.items()
            if k not in drop and not any(k.startswith(p + ".") for p in drop)
        }

    def parameters(self) -> dict[str, Any]:
        return dict(self.trainable_parameters())
