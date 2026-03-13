"""Klear model architecture — MoE with optional MLP-only layers and shared experts.

Implements ModelProtocol. Ported from mlx_lm. Uses RoPE, Q/K RMSNorm,
SwiGLU MLP, and sparse MoE (SwitchGLU) with shared experts and expert bias.
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
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Klear model configuration."""

    model_type: str = "klear"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    attention_bias: bool = False
    mlp_only_layers: list[int] | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    decoder_sparse_step: int = 2
    n_shared_experts: int = 2
    moe_intermediate_size: int = 2048
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    num_key_value_heads: int | None = None
    rope_theta: float = 10000.0
    max_position_embeddings: int = 131072
    norm_topk_prob: bool = True

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []


class KlearAttention(nn.Module):
    """Multi-head attention with Q/K RMSNorm and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
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

        queries = self.q_norm(queries.reshape(B, L, self.num_attention_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

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
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class KlearMLP(nn.Module):
    """SwiGLU MLP (gate + up -> swiglu -> down)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class KlearSparseMoeBlock(nn.Module):
    """Sparse MoE block: gate, routed experts (SwitchGLU), shared experts, coefficient."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.experts = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
        self.shared_experts = KlearMLP(
            args.hidden_size,
            hidden_dim=args.moe_intermediate_size * args.n_shared_experts,
        )
        self.coefficient = nn.Linear(args.hidden_size, 2)
        self.expert_bias = mx.zeros((self.num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        routing_weights = mx.sigmoid(self.gate(x).astype(mx.float32))
        biased_weights = routing_weights + self.expert_bias.reshape((1, 1, -1))
        k = self.top_k
        inds = mx.argpartition(-biased_weights, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(routing_weights, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)
        expert_out = self.experts(x, inds)
        y_experts = (expert_out * scores[..., None]).sum(axis=-2)
        coef = mx.softmax(self.coefficient(x), axis=-1, precise=True)
        shared = self.shared_experts(x)
        return y_experts * coef[..., :1] + shared * coef[..., 1:]


class KlearDecoderLayer(nn.Module):
    """Decoder layer: input norm -> attention -> residual -> post norm -> MLP/MoE -> residual."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = KlearAttention(args)

        use_moe = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        if use_moe:
            self.mlp = KlearSparseMoeBlock(args)
        else:
            self.mlp = KlearMLP(args.hidden_size, args.intermediate_size)

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
        return h + self.mlp(self.post_attention_layernorm(h))


class KlearModel(nn.Module):
    """Klear transformer backbone: embed -> decoder layers -> final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KlearDecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
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
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    """Klear LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KlearModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
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
        if "model.layers.0.mlp.experts.0.gate_proj.weight" not in weights:
            return weights

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp.experts"
            for name in ("gate_proj", "up_proj", "down_proj"):
                stacked = [
                    weights.pop(f"{prefix}.{e}.{name}.weight")
                    for e in range(self.args.num_experts)
                ]
                weights[f"{prefix}.{name}.weight"] = mx.stack(stacked)

        return weights
