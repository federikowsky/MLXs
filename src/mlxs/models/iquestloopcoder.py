"""IQuest Loop Coder model: dual-loop attention with gate-mixed global/local (§7).

Port from mlx_lm. Implements ModelProtocol. Uses RMSNorm, SwiGLU MLP, RoPE;
loop_num=2 with full-attention pass then gate-mixed global (full) + local (sliding
window) pass. Cache: KVCache per layer for first pass, RotatingKVCache for second.
Imports: mlxs.cache, mlxs.layers, mlxs.models.base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache import KVCache, RotatingKVCache
from mlxs.cache.attention_mask import create_attention_mask
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


def _compute_gate(query: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """Gate logits from query: (B, n_heads, L, head_dim) -> (B, n_heads, L)."""
    # query (B, n_heads, L, head_dim), weight (n_heads, head_dim), bias (n_heads,)
    gate_logits = (query * weight[None, :, None, :]).sum(axis=-1) + bias[None, :, None]
    return mx.sigmoid(gate_logits)


def _mix_attention(gate: mx.array, attn_global: mx.array, attn_local: mx.array) -> mx.array:
    """Mix global and local attention: gate * global + (1 - gate) * local."""
    gate = gate[..., None]  # (B, n_heads, L) -> (B, n_heads, L, 1) for broadcast
    return gate * attn_global + (1.0 - gate) * attn_local


@dataclass
class ModelArgs(BaseModelArgs):
    """IQuest Loop Coder configuration. Only loop_num=2 is supported."""

    model_type: str = "iquestloopcoder"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    head_dim: int = 128
    num_key_value_heads: int = 32
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = False
    loop_num: int = 2
    loop_window_size: int = 64


class LoopGateProjection(nn.Module):
    """Per-head gate from query for mixing global vs local attention."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = mx.zeros((num_heads, head_dim))
        self.bias = mx.zeros((num_heads,))

    def __call__(self, query: mx.array) -> mx.array:
        return _compute_gate(query, self.weight, self.bias)


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def get_qkv(self, x: mx.array, offset: int = 0) -> tuple[mx.array, mx.array, mx.array]:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        return queries, keys, values

    def attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        bias = args.mlp_bias
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm block: attention + MLP with residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


class IQuestLoopCoderModel(nn.Module):
    """Backbone: embed + dual-loop transformer layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        if args.loop_num != 2:
            raise ValueError(f"Only loop_num=2 is supported, got {args.loop_num}")
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.gate_projections = [
            LoopGateProjection(args.num_attention_heads, args.head_dim)
            for _ in range(args.num_hidden_layers)
        ]
        self.loop_num = args.loop_num
        self.loop_window_size = args.loop_window_size

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        B, L = inputs.shape[:2]
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * (2 * len(self.layers))  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0])
        window_mask = create_attention_mask(
            h, cache[len(self.layers)], window_size=self.loop_window_size
        )

        loop1_kv: list[tuple[mx.array, mx.array]] = []
        first_caches = cache[: len(self.layers)]
        for layer, c in zip(self.layers, first_caches, strict=True):
            h_norm = layer.input_layernorm(h)
            offset = c.offset if c is not None else 0
            q1, k1, v1 = layer.self_attn.get_qkv(h_norm, offset)
            if c is not None:
                k1, v1 = c.update_and_fetch(k1, v1)
            loop1_kv.append((k1, v1))

            out = layer.self_attn.attention(q1, k1, v1, mask, cache=c)
            r = layer.self_attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        for layer, gate_proj, c, (k1, v1) in zip(
            self.layers,
            self.gate_projections,
            cache[len(self.layers) :],
            loop1_kv,
            strict=True,
        ):
            h_norm = layer.input_layernorm(h)
            offset = c.offset if c is not None else 0
            q2, k2, v2 = layer.self_attn.get_qkv(h_norm, offset)
            gate = gate_proj(q2)
            attn_global = layer.self_attn.attention(q2, k1, v1, mask, cache=c)

            if c is not None:
                k2, v2 = c.update_and_fetch(k2, v2)
            attn_local = layer.self_attn.attention(q2, k2, v2, window_mask, cache=c)

            mixed = _mix_attention(gate, attn_global, attn_local)
            r = layer.self_attn.o_proj(mixed.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        return self.norm(h)


class Model(nn.Module):
    """IQuest Loop Coder — satisfies ModelProtocol (§7.3, AC17)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = IQuestLoopCoderModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def num_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(
        self,
    ) -> list[KVCache | RotatingKVCache]:
        n = len(self.model.layers)
        return [KVCache() for _ in range(n)] + [
            RotatingKVCache(max_size=self.args.loop_window_size) for _ in range(n)
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        weights = {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def parameters(self) -> dict[str, Any]:
        return dict(self.items())
