"""Qwen3 VL MoE: vision-language wrapper around Qwen3 MoE text backbone.

Text-only path: forward with input_ids and cache returns logits (B,T,V).
Implements ModelProtocol. Weights: drop visual, remap language_model + expert stacking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache

# Lazy import to keep DAG: models depend on protocols, not each other.
# qwen3_moe is the text backbone (same as mlx_lm qwen3_vl_moe -> qwen3_moe).
from mlxs.models import qwen3_moe
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen3 VL MoE config: model_type + text_config (same as mlx_lm)."""

    model_type: str = "qwen3_vl_moe"
    text_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        if "text_config" not in params:
            return cls(
                model_type=params.get("model_type", "qwen3_vl_moe"),
                text_config=params,
            )
        return cls(
            model_type=params.get("model_type", "qwen3_vl_moe"),
            text_config=params.get("text_config"),
        )


class Model(nn.Module):
    """Qwen3 VL MoE: wraps Qwen3 MoE text backbone; satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        if args.text_config is None:
            raise ValueError("qwen3_vl_moe requires text_config")
        self.args = args
        self.model_type = args.model_type
        text_args = qwen3_moe.ModelArgs.from_dict(args.text_config)
        self.language_model = qwen3_moe.Model(text_args)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(inputs, cache=cache)

    @property
    def num_layers(self) -> int:
        return self.language_model.num_layers

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    def make_cache(self) -> list[KVCache]:
        return self.language_model.make_cache()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        # Drop vision/visual (VL checkpoint); keep only language_model.
        out = {
            k: v
            for k, v in weights.items()
            if not k.startswith("visual")
            and not k.startswith("vision_tower")
            and not k.startswith("model.visual")
        }
        # Ensure language_model.* keys for our module layout.
        normalized: dict[str, Any] = {}
        for k, v in out.items():
            if not k.startswith("language_model."):
                normalized["language_model." + k] = v
            else:
                normalized[k] = v
        out = normalized
        # Expert stacking: language_model.model.layers.{l}.mlp.experts.{e}.{n} -> switch_mlp
        num_hidden_layers = self.language_model.args.num_hidden_layers
        num_experts = self.language_model.args.num_experts
        prefix = "language_model.model.layers"
        for layer_idx in range(num_hidden_layers):
            layer_prefix = f"{prefix}.{layer_idx}.mlp"
            gate_up_key = f"{layer_prefix}.experts.0.gate_proj.weight"
            if gate_up_key not in out and f"{layer_prefix}.experts.0.up_proj.weight" not in out:
                continue
            for n in ["up_proj", "down_proj", "gate_proj"]:
                key_template = f"{layer_prefix}.experts.{{e}}.{n}.weight"
                if key_template.format(e=0) not in out:
                    continue
                to_join = [
                    out.pop(f"{layer_prefix}.experts.{e}.{n}.weight") for e in range(num_experts)
                ]
                out[f"{layer_prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)
        # Inner tie_word_embeddings
        if self.language_model.args.tie_word_embeddings:
            out.pop("language_model.lm_head.weight", None)
        return out
