"""Cohere Command model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports parallel attention+MLP blocks, optional QK norm, logit scaling,
and tied embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache
from mlxs.models.activations import swiglu
from mlxs.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)


@dataclass
class ModelArgs(BaseModelArgs):
    """Cohere model configuration."""

    model_type: str = "cohere"
    hidden_size: int = 8192
    num_hidden_layers: int = 40
    intermediate_size: int = 22528
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    rope_theta: float = 8000000.0
    vocab_size: int = 256000
    layer_norm_eps: float = 1e-05
    logit_scale: float = 0.0625
    attention_bias: bool = False
    layer_norm_bias: bool = False
    use_qk_norm: bool = False


class LayerNorm2D(nn.Module):
    """Per-head layer norm for QK normalization."""

    def __init__(self, d1: int, d2: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.zeros((d1, d2))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps)


class Attention(nn.Module):
    """Multi-head attention with optional QK norm and traditional RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // args.num_attention_heads
        self.scale = head_dim**-0.5

        attention_bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = LayerNorm2D(self.n_heads, head_dim, eps=args.layer_norm_eps)
            self.k_norm = LayerNorm2D(
                self.n_kv_heads, head_dim, eps=args.layer_norm_eps
            )

        self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
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
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Parallel attention + MLP block with shared LayerNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x


class CohereModel(nn.Module):
    """Cohere transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )

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
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    """Cohere LM head wrapper — satisfies ModelProtocol.

    Uses tied embeddings with logit scaling.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = CohereModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        return out * self.args.logit_scale

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.layers
