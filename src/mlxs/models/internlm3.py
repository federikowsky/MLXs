"""InternLM3 model — port from mlx_lm, ModelProtocol-compliant.

Uses separate q/k/v/o projections, RMSNorm, SwiGLU MLP, Dynamic NTK RoPE.
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
from mlxs.layers.rope import DynamicNTKScalingRoPE
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """InternLM3 config; rope_scaling uses 'factor' and 'rope_type' (mlx_lm)."""

    model_type: str = "internlm3"
    hidden_size: int = 2048
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 92544
    bias: bool = False
    qkv_bias: bool = False
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
            required = {"factor", "rope_type"}
            # Accept "type" as alias for "rope_type" (e.g. HuggingFace configs)
            scaling = dict(self.rope_scaling)
            if "type" in scaling and "rope_type" not in scaling:
                scaling["rope_type"] = scaling["type"]
            if not all(k in scaling for k in required):
                raise ValueError(f"rope_scaling must contain {required}")
            if scaling["rope_type"] not in ("linear", "dynamic"):
                raise ValueError("rope_scaling rope_type must be 'linear' or 'dynamic'")


class Attention(nn.Module):
    """Multi-head attention with separate q/k/v/o and Dynamic NTK RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5
        qkv_bias = args.qkv_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=qkv_bias)
        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None
            and args.rope_scaling.get("rope_type", args.rope_scaling.get("type")) == "linear"
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


class MLP(nn.Module):
    """SwiGLU MLP: gate_proj, up_proj, down_proj."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm block: input_layernorm -> attention, post_attention_layernorm -> MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size, args.bias)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class InternLM3Model(nn.Module):
    """InternLM3 transformer stack: embed_tokens, layers, norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0] if cache else None)
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """InternLM3 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = InternLM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
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
