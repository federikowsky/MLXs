"""Qwen2 / Qwen3 model architecture (§7.1 Phase 1, AC17).

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports GQA, RoPE, SwiGLU, and sliding window attention.

Validates registry extensibility: this file + registry entry is all
that's needed to add Qwen support. No changes to generate, cache,
batch, or server.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache
from mlxs.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)


@partial(mx.compile, shapeless=True)
def _swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen2/3 model configuration (§7.2)."""

    model_type: str = "qwen2"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    num_key_value_heads: int | None = None
    attention_bias: bool = True  # Qwen uses attention bias by default
    mlp_bias: bool = False
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = False  # Qwen2 typically doesn't tie
    sliding_window: int | None = None
    use_sliding_window: bool = False
    max_window_layers: int = 0  # Layers that use sliding window (from bottom)

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


def _initialize_rope(
    dims: int,
    base: float = 10000.0,
    traditional: bool = False,
    scaling: dict[str, float | str] | None = None,
) -> nn.RoPE:
    """Initialize RoPE with optional scaling."""
    if scaling is None:
        return nn.RoPE(dims, traditional=traditional, base=base)

    rope_type = scaling.get("rope_type", scaling.get("type", "default"))

    if rope_type == "linear":
        scale = 1 / scaling["factor"]
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    if rope_type == "yarn":
        scale = 1 / scaling.get("factor", 1.0)
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    return nn.RoPE(dims, traditional=traditional, base=base)


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE for Qwen."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        bias = args.attention_bias

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Sliding window for lower layers
        self.use_sliding = (
            args.use_sliding_window
            and args.sliding_window is not None
            and layer_idx < args.max_window_layers
        )
        self.sliding_window = args.sliding_window if self.use_sliding else None

        self.rope = _initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """SwiGLU MLP for Qwen."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden = args.intermediate_size
        bias = args.mlp_bias
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block for Qwen."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx=layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class QwenModel(nn.Module):
    """Qwen transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

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
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """Qwen LM head wrapper — satisfies ModelProtocol (§7.3, AC17)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = QwenModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        # Remove rotary embedding inverse frequencies
        weights = {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.layers
