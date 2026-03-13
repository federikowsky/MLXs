"""Lille 130M model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
RoPE, GQA, RMSNorm, SwiGLU MLP. Tied embeddings for LM head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Lille 130M model configuration."""

    model_type: str = "lille_130m"
    block_size: int = 2048
    layer_norm_eps: float = 1e-5
    n_embd: int = 768
    n_head: int = 12
    n_kv_heads: int = 12
    n_layer: int = 24
    rope_theta: float = 10000.0
    vocab_size: int = 32000
    tie_word_embeddings: bool = True


class Lille130mAttention(nn.Module):
    """Multi-head attention with RoPE and optional GQA."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_head = args.n_head
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.n_embd // args.n_head
        self.scale = self.head_dim**-0.5

        qkv_size = (args.n_head + 2 * args.n_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(args.n_embd, qkv_size, bias=False)
        self.out_proj = nn.Linear(args.n_head * self.head_dim, args.n_embd, bias=False)
        self.norm = nn.RMSNorm(args.n_embd, eps=args.layer_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.qkv_proj(self.norm(x))
        q_size = self.n_head * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        queries, keys, values = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
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
        return self.out_proj(output)


class Lille130mMLP(nn.Module):
    """SwiGLU MLP with RMSNorm before gate/up."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = 256 * round(int(8 * args.n_embd / 3) / 256)
        self.norm = nn.RMSNorm(args.n_embd, eps=args.layer_norm_eps)
        self.gate_proj = nn.Linear(args.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(args.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, args.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        return self.down_proj(swiglu(self.gate_proj(h), self.up_proj(h)))


class Lille130Block(nn.Module):
    """Single transformer block: attention + MLP with residuals."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attention = Lille130mAttention(args)
        self.feed_forward = Lille130mMLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.attention(x, mask, cache)
        return h + self.feed_forward(h)


class Lille130Model(nn.Module):
    """Lille 130M transformer backbone with RoPE and tied LM head."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.n_embd)
        self.layers = [Lille130Block(args) for _ in range(args.n_layer)]
        self.norm = nn.RMSNorm(args.n_embd, eps=args.layer_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.tok_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """Lille 130M LM wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = Lille130Model(args)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        hidden = self.transformer(input_ids, cache=cache)
        return self.transformer.tok_embeddings.as_linear(hidden)

    @property
    def num_layers(self) -> int:
        return len(self.transformer.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.transformer.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in weights.items() if "rotary_emb" not in k}

    @property
    def layers(self) -> list[Lille130Block]:
        return self.transformer.layers
