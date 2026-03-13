"""OLMo 3 model — port from mlx_lm (mlx_lm/models/olmo3.py), ModelProtocol-compliant.

Hybrid: sliding-window attention (RotatingKVCache) and full attention (KVCache)
per layer_types. QK-norm, post-norm blocks, optional RoPE scaling on full layers.
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
    """OLMo 3 config (config.json)."""

    model_type: str = "olmo3"
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 50304
    max_position_embeddings: int = 8192
    sliding_window: int = 4096
    rope_theta: float = 10000.0
    attention_bias: bool = False
    layer_types: list[str] | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]


class Attention(nn.Module):
    """OLMo 3 attention: QK-norm, per-layer RoPE (sliding vs full with optional scaling)."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.n_heads * self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.n_kv_heads * self.head_dim, eps=args.rms_norm_eps)

        layer_type = args.layer_types[layer_idx]
        if layer_type != "full_attention":
            self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)
        else:
            self.rope = initialize_rope(
                self.head_dim,
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
        queries = self.q_norm(self.q_proj(x))
        keys = self.k_norm(self.k_proj(x))
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

        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """SwiGLU MLP (no bias in mlx_lm olmo3)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Post-norm block: norm after attn and after MLP, then add residual."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        r = self.post_attention_layernorm(self.self_attn(x, mask, cache))
        h = x + r
        r = self.post_feedforward_layernorm(self.mlp(h))
        return h + r


class Olmo3Model(nn.Module):
    """OLMo 3 transformer: embed, mixed sliding/full layers, final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.layer_types = args.layer_types
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Find first index of each layer type for mask creation
        self.ga_idx = 0
        self.swa_idx = 0
        for i, lt in enumerate(self.layer_types):
            if lt == "full_attention":
                self.ga_idx = i
                break
        for i, lt in enumerate(self.layer_types):
            if lt != "full_attention":
                self.swa_idx = i
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        full_mask = create_attention_mask(h, cache[self.ga_idx])
        sliding_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.sliding_window)

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types, strict=True):
            mask = full_mask if layer_type == "full_attention" else sliding_mask
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """OLMo 3 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Olmo3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)  # type: ignore[assignment]
        return out

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(
        self,
    ) -> list[KVCache | RotatingKVCache]:
        return [
            KVCache()
            if lt == "full_attention"
            else RotatingKVCache(max_size=self.args.sliding_window)
            for lt in self.model.layer_types
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in weights.items() if "inv_freq" not in k}
