"""GLM-4 model — port from mlx_lm (mlx_lm/models/glm4.py), ModelProtocol-compliant.

Dense decoder-only: RMSNorm, SwiGLU MLP, partial RoPE (first head_dim * partial_rotary_factor).
Imports only from mlxs.cache, mlxs.layers, mlxs.models.base.
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
    """GLM-4 config (config.json)."""

    model_type: str = "glm4"
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    attention_bias: bool = False
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 16
    partial_rotary_factor: float = 1.0
    rope_theta: float = 10000.0
    rope_traditional: bool = True
    max_position_embeddings: int = 32768


class Glm4MLP(nn.Module):
    """SwiGLU MLP with combined gate_up projection."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(args.hidden_size, 2 * args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.gate_up_proj(x)
        gate, up_states = mx.split(x, 2, axis=-1)
        return self.down_proj(swiglu(gate, up_states))


class Glm4Attention(nn.Module):
    """Multi-head attention with partial RoPE (first head_dim * partial_rotary_factor dims)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.head_dim = getattr(args, "head_dim", args.hidden_size // args.num_attention_heads)
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        rope_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope = nn.RoPE(
            dims=rope_dims,
            base=args.rope_theta,
            traditional=args.rope_traditional,
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
        return self.o_proj(output)


class Glm4DecoderLayer(nn.Module):
    """Decoder block: input norm → attention → post_attn norm → MLP → post_mlp norm, residuals."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Glm4Attention(args=args)
        self.mlp = Glm4MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_self_attn_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        x = x + self.post_self_attn_layernorm(self.self_attn(self.input_layernorm(x), mask, cache))
        residual = x
        x = self.post_mlp_layernorm(self.mlp(self.post_attention_layernorm(x))) + residual
        return x


class Glm4Model(nn.Module):
    """GLM-4 transformer: embed, decoder layers, final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Glm4DecoderLayer(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        if mask is None:
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """GLM-4 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache, mask=mask)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.num_layers)]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights
