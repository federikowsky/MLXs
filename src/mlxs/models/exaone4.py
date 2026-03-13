"""ExaOne4 model — port from mlx_lm (mlx_lm/models/exaone4.py), ModelProtocol-compliant.

Dense decoder with optional sliding-window attention: QK norm, RoPE, SwiGLU MLP,
post-attention / post-FFN RMSNorm. Local layers use RotatingKVCache; global use KVCache.
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
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """ExaOne4 config (config.json)."""

    model_type: str = "exaone4"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 16
    max_position_embeddings: int = 131072
    rope_theta: float = 10000.0
    head_dim: int = 128
    tie_word_embeddings: bool = True
    rope_scaling: dict[str, float | str] | None = None
    sliding_window: int | None = None
    sliding_window_pattern: list[str] | None = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_local: bool | None) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.is_local = is_local if is_local is not None else False
        self.use_rope = is_local is None or is_local
        if self.use_rope:
            self.rope = initialize_rope(
                head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            if self.use_rope:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        elif self.use_rope:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, is_local: bool | None) -> None:
        super().__init__()
        self.self_attn = Attention(args, is_local)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(x, mask, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(h)
        return h + self.post_feedforward_layernorm(r)


class Exaone4Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        pattern = args.sliding_window_pattern
        self.layers = [
            TransformerBlock(
                args,
                is_local=pattern[i % len(pattern)] == "L" if pattern else None,
            )
            for i in range(args.num_hidden_layers)
        ]
        if pattern:
            self.swa_idx: int | None = pattern.index("L")
            self.full_idx = pattern.index("G")
        else:
            self.swa_idx = None
            self.full_idx = 0
        self.window_size = args.sliding_window
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        global_mask = create_attention_mask(h, cache[self.full_idx])
        if self.swa_idx is not None and self.window_size is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.window_size
            )
        else:
            swa_mask = None
        for layer, c in zip(self.layers, cache, strict=True):
            mask = swa_mask if layer.self_attn.is_local else global_mask
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """ExaOne4 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = Exaone4Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.transformer(input_ids, cache)
        if self.args.tie_word_embeddings:
            out = self.transformer.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)  # type: ignore[assignment]
        return out

    @property
    def num_layers(self) -> int:
        return len(self.transformer.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        window_size = self.args.sliding_window or 0
        pattern = self.args.sliding_window_pattern
        return [
            (
                RotatingKVCache(max_size=window_size, keep=0)
                if pattern and pattern[i % len(pattern)] == "L"
                else KVCache()
            )
            for i in range(len(self.transformer.layers))
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights
