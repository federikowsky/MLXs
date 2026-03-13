"""InternLM2 model — port from mlx_lm, ModelProtocol-compliant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.rope import DynamicNTKScalingRoPE
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "internlm2"
    hidden_size: int = 2048
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 92544
    bias: bool = True
    max_position_embeddings: int = 32768
    num_key_value_heads: int | None = None
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.rope_scaling is not None:
            required = {"factor", "type"}
            if not all(k in self.rope_scaling for k in required):
                raise ValueError(f"rope_scaling must contain {required}")
            if self.rope_scaling["type"] not in ("linear", "dynamic"):
                raise ValueError("rope_scaling type must be 'linear' or 'dynamic'")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = args.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        self.wqkv = nn.Linear(
            dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=args.bias
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.bias)
        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 2.0
        )
        self.rope = DynamicNTKScalingRoPE(
            self.head_dim,
            max_position_embeddings=args.max_position_embeddings,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        qkv_states = self.wqkv(x)
        qkv_states = qkv_states.reshape(B, L, -1, 2 + self.n_kv_groups, self.head_dim)
        queries = qkv_states[..., : self.n_kv_groups, :]
        queries = queries.reshape(B, L, -1, self.head_dim)
        keys = qkv_states[..., -2, :]
        values = qkv_states[..., -1, :]
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
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
        return self.wo(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size)
        self.attention_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r


class InternLM2Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.tok_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0] if cache else None)
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = InternLM2Model(args)
        if not args.tie_word_embeddings:
            self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        if self.args.tie_word_embeddings:
            out = self.model.tok_embeddings.as_linear(out)
        else:
            out = self.output(out)
        return out

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(len(self.model.layers))]

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in weights.items() if "attention.rope.inv_freq" not in k}
