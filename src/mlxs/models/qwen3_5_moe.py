"""Qwen3.5 MoE model: hybrid linear + attention layers with sparse MoE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.moe import SwitchGLU
from mlxs.models.base import BaseModelArgs
from mlxs.models.qwen3_5 import (
    MLP,
    Attention,
    GatedDeltaNet,
)
from mlxs.models.qwen3_5 import (
    Model as Qwen35LanguageModel,
)
from mlxs.models.qwen3_5 import (
    ModelArgs as Qwen35ModelArgs,
)


@dataclass
class TextModelArgs(Qwen35ModelArgs):
    """Qwen3.5-MoE text config from ``text_config``."""

    model_type: str = "qwen3_5_moe"
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    mlp_only_layers: list[int] = field(default_factory=list)
    rope_parameters: dict[str, Any] | None = field(
        default_factory=lambda: {
            "type": "default",
            "rope_theta": 100000.0,
            "partial_rotary_factor": 0.25,
        }
    )


class SparseMoeBlock(nn.Module):
    """Sparse MoE with top-k routing, SwitchGLU experts, and one shared expert."""

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        shared_size = args.shared_expert_intermediate_size or args.intermediate_size

        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, self.num_experts)
        self.shared_expert = MLP(dim, shared_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        routed = self.switch_mlp(x, inds)
        routed = (routed * scores[..., None]).sum(axis=-2)

        shared = self.shared_expert(x)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
        return routed + shared


class DecoderLayer(nn.Module):
    """Hybrid decoder layer with dense or sparse feed-forward block."""

    def __init__(self, args: TextModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        is_moe = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.mlp: SparseMoeBlock | MLP = (
            SparseMoeBlock(args) if is_moe else MLP(args.hidden_size, args.intermediate_size)
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        if self.is_linear:
            residual = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            residual = self.self_attn(self.input_layernorm(x), mask, cache)
        hidden = x + residual
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TextModel(nn.Module):
    """Qwen3.5-MoE text backbone."""

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._ssm_idx = 0
        self._fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        fa_mask = create_attention_mask(hidden, cache[self._fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[self._ssm_idx])
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        return self.norm(hidden)


@dataclass
class ModelArgs(BaseModelArgs):
    """Top-level Qwen3.5-MoE config: ``model_type`` + ``text_config``."""

    model_type: str = "qwen3_5_moe"
    text_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        if "text_config" not in params:
            return cls(
                model_type=params.get("model_type", "qwen3_5_moe"),
                text_config=params,
            )
        return cls(**{k: v for k, v in params.items() if k in ("model_type", "text_config")})


class Model(nn.Module):
    """Qwen3.5-MoE wrapper that reuses the dense language-model head/cache logic."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = TextModelArgs.from_dict(args.text_config)
        self.language_model = Qwen35LanguageModel(text_args)
        self.language_model.model = TextModel(text_args)
        self.language_model.args = text_args

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
        )

    @property
    def num_layers(self) -> int:
        return self.language_model.num_layers

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return self.language_model.make_cache()

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.language_model.layers

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        sanitized = _normalize_language_model_weights(weights)
        for layer_idx in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer_idx}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key not in sanitized:
                continue

            gate_up = sanitized.pop(gate_up_key)
            mid = gate_up.shape[-2] // 2
            sanitized[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
            sanitized[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]

            down_key = f"{prefix}.experts.down_proj"
            if down_key in sanitized:
                sanitized[f"{prefix}.switch_mlp.down_proj.weight"] = sanitized.pop(down_key)

        return _text_model_sanitize(self.language_model, sanitized)


def _normalize_language_model_weights(weights: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in weights.items():
        if key.startswith(("vision_tower", "visual.", "model.visual")):
            continue
        if key.startswith("model.language_model"):
            key = key.replace("model.language_model", "language_model.model", 1)
        elif not key.startswith("language_model."):
            key = "language_model." + key
        normalized[key] = value
    return normalized


def _text_model_sanitize(
    language_model: Qwen35LanguageModel,
    weights: dict[str, Any],
) -> dict[str, Any]:
    return language_model.sanitize(weights)
