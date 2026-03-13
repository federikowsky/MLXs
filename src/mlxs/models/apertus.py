"""Apertus model — port from mlx_lm (ModelProtocol-compliant).

Dense decoder: GQA, RoPE (optional scaling), XieLU MLP, RMSNorm, optional QK-norm.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import XieLU
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Apertus config (config.json)."""

    model_type: str = "apertus"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    intermediate_size: int = 8192
    mlp_bias: bool = False
    num_attention_heads: int = 16
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    num_key_value_heads: int = 16
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    post_norm: bool = True
    qk_norm: bool = True
    tie_word_embeddings: bool = False
    rope_traditional: bool = False
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        sig = inspect.signature(cls)
        return cls(**{k: v for k, v in params.items() if k in sig.parameters})


class ApertusMLP(nn.Module):
    """MLP: up_proj -> XieLU -> down_proj."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)
        self.act_fn = XieLU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class ApertusAttention(nn.Module):
    """GQA with optional QK-norm and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.num_attention_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class ApertusDecoderLayer(nn.Module):
    """Pre-norm: norm -> attn -> residual, norm -> mlp -> residual."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = ApertusAttention(args)
        self.mlp = ApertusMLP(args)
        self.attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.feedforward_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.attention_layernorm(x), mask, cache)
        return h + self.mlp(self.feedforward_layernorm(h))


class ApertusModel(nn.Module):
    """Embed + decoder layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ApertusDecoderLayer(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """Apertus LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ApertusModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        for k, v in list(weights.items()):
            if k.endswith("alpha_p") or k.endswith("alpha_n"):
                weights[k] = v.squeeze()
        return weights
