"""ExaOne model — port from mlx_lm (mlx_lm/models/exaone.py), ModelProtocol-compliant.

Dense decoder: standard GQA, RoPE, SwiGLU MLP, RMSNorm.
Config uses num_layers and layer_norm_epsilon.
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
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """ExaOne config (config.json)."""

    model_type: str = "exaone"
    hidden_size: int = 2048
    num_layers: int = 24
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    vocab_size: int = 151936
    rope_theta: float = 10000.0
    layer_norm_epsilon: float = 1e-6
    num_key_value_heads: int = 16
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    rope_traditional: bool = False
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False


class Attention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or dim // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden = args.intermediate_size
        self.c_fc_0 = nn.Linear(dim, hidden, bias=args.mlp_bias)
        self.c_fc_1 = nn.Linear(dim, hidden, bias=args.mlp_bias)
        self.c_proj = nn.Linear(hidden, dim, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(swiglu(self.c_fc_0(x), self.c_fc_1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.attn = Attention(args)
        self.ln_2 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.attn(self.ln_1(x), mask, cache)
        return h + self.mlp(self.ln_2(h))


class ExaoneModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.wte(inputs)
        if cache is None:
            cache = [None] * len(self.h)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.h, cache, strict=True):
            h = layer(h, mask, cache=c)
        return self.ln_f(h)


class Model(nn.Module):
    """ExaOne LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = ExaoneModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.transformer(input_ids, cache)
        if self.args.tie_word_embeddings:
            out = self.transformer.wte.as_linear(out)
        else:
            out = self.lm_head(out)  # type: ignore[assignment]
        return out

    @property
    def num_layers(self) -> int:
        return len(self.transformer.h)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.transformer.h]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights
