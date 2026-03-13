"""GPT-BigCode model architecture (StarCoder v1).

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports multi-query attention with learned position embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache
from mlxs.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)


@dataclass
class ModelArgs(BaseModelArgs):
    """GPT-BigCode model configuration."""

    model_type: str = "gpt_bigcode"
    n_embd: int = 2048
    n_layer: int = 24
    n_inner: int = 8192
    n_head: int = 16
    n_positions: int = 8192
    layer_norm_epsilon: float = 1e-5
    vocab_size: int = 49152
    num_key_value_heads: int | None = None
    multi_query: bool = True
    attention_bias: bool = True
    mlp_bias: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = 1 if self.multi_query else self.n_head


class Attention(nn.Module):
    """Multi-query / multi-head attention with fused QKV projection."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.n_embd
        self.n_heads = args.n_head
        self.n_kv_heads = 1 if args.multi_query else args.n_head
        self.head_dim = dim // self.n_heads
        self.dim = dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        bias = args.attention_bias
        self.c_attn = nn.Linear(dim, dim + 2 * self.kv_dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(
            qkv, [self.dim, self.dim + self.kv_dim], axis=-1
        )

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.c_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """GELU MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        bias = args.mlp_bias
        self.c_fc = nn.Linear(args.n_embd, args.n_inner, bias=bias)
        self.c_proj = nn.Linear(args.n_inner, args.n_embd, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.ln_1 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.attn(self.ln_1(x), mask, cache)
        return h + self.mlp(self.ln_2(h))


class GPTBigCodeModel(nn.Module):
    """GPT-BigCode transformer backbone with learned position embeddings."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.n_positions, args.n_embd)
        self.h = [TransformerBlock(args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        _, L = inputs.shape

        hidden_states = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.h)  # type: ignore[list-item]
            position_ids = mx.arange(L)
        else:
            offset = cache[0].offset if cache[0] is not None else 0
            position_ids = mx.arange(offset, offset + L)

        mask = create_attention_mask(hidden_states, cache[0])
        hidden_states = hidden_states + self.wpe(position_ids)

        for layer, c in zip(self.h, cache, strict=True):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):
    """GPT-BigCode LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = GPTBigCodeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.transformer(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.transformer.wte.as_linear(out)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.transformer.h)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.transformer.h]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.transformer.h
