"""Cohere2 model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports sliding window attention with configurable pattern, parallel
attention+MLP blocks, and logit scaling with tied embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Cohere2 model configuration."""

    model_type: str = "cohere2"
    hidden_size: int = 4096
    head_dim: int = 128
    num_hidden_layers: int = 32
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rope_theta: float = 50000.0
    vocab_size: int = 256000
    layer_norm_eps: float = 1e-05
    logit_scale: float = 0.0625
    attention_bias: bool = False
    layer_norm_bias: bool = False
    sliding_window: int = 4096
    sliding_window_pattern: int = 4


class Attention(nn.Module):
    """Multi-head attention with conditional RoPE and sliding window support."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        if (head_dim * n_heads) != dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {dim}"
                f" and `num_heads`: {n_heads})."
            )
        self.scale = head_dim**-0.5

        attention_bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

        # Sliding window layers use RoPE; global layers do not.
        self.use_sliding_window = (layer_idx + 1) % args.sliding_window_pattern != 0

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE only for sliding window layers.
        if self.use_sliding_window:
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        sdpa_type = mx.float32 if queries.dtype == mx.float16 else queries.dtype
        output = scaled_dot_product_attention(
            queries.astype(sdpa_type),
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        ).astype(queries.dtype)
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

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x


class Cohere2Model(nn.Module):
    """Cohere2 transformer backbone with sliding window pattern."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.window_size = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        j = self.args.sliding_window_pattern
        full_mask = create_attention_mask(h, cache[j - 1])
        swa_mask = create_attention_mask(h, cache[0], window_size=self.window_size)

        for i, (layer, c) in enumerate(zip(self.layers, cache, strict=True)):
            is_global = (
                i % self.args.sliding_window_pattern
                == self.args.sliding_window_pattern - 1
            )
            mask = full_mask if is_global else swa_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    """Cohere2 LM head wrapper — satisfies ModelProtocol.

    Uses tied embeddings with logit scaling and mixed
    KVCache / RotatingKVCache per the sliding window pattern.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Cohere2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
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

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        caches: list[KVCache | RotatingKVCache] = []
        for i in range(self.args.num_hidden_layers):
            if (
                i % self.args.sliding_window_pattern
                == self.args.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
        return caches

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.layers
