"""PhiMoE model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports GQA, SuScaledRoPE, and Mixture-of-Experts with SwitchGLU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import SuScaledRoPE
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """PhiMoE model configuration."""

    model_type: str = "phimoe"
    vocab_size: int = 32064
    hidden_size: int = 4096
    intermediate_size: int = 6400
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_scaling: dict[str, float | list[float]] | None = None
    num_local_experts: int = 16
    num_experts_per_tok: int = 2
    rope_theta: float = 10000.0


class Attention(nn.Module):
    """Multi-head attention with GQA and SuScaledRoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = dim // self.n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=True)

        self.rope = SuScaledRoPE(
            head_dim,
            base=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            original_max_position_embeddings=args.original_max_position_embeddings,
            short_factor=args.rope_scaling["short_factor"],
            long_factor=args.rope_scaling["long_factor"],
            short_mscale=args.rope_scaling["short_mscale"],
            long_mscale=args.rope_scaling["long_mscale"],
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
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
            queries, keys, values, cache=cache, scale=self.scale, mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class PhiMoESparseMoeBlock(nn.Module):
    """Top-k gated MoE block with SwitchGLU experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.intermediate_size
        self.num_experts = args.num_local_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(self.hidden_dim, self.ffn_dim, self.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        k = self.top_k
        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class PhiMoEDecoderLayer(nn.Module):
    """Pre-norm transformer block with MoE MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.block_sparse_moe = PhiMoESparseMoeBlock(args)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        residual = x
        hidden_states = self.input_layernorm(x)
        hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        return residual + hidden_states


class PhiMoEModel(nn.Module):
    """PhiMoE transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            PhiMoEDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    """PhiMoE LM head wrapper — satisfies ModelProtocol (AC17)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = PhiMoEModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=True)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache)
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
        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.block_sparse_moe.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(
                                f"{prefix}.block_sparse_moe.experts.{e}.{n}.{k}"
                            )
                            for e in range(self.args.num_local_experts)
                        ]
                        weights[
                            f"{prefix}.block_sparse_moe.switch_mlp.{m}.{k}"
                        ] = mx.stack(to_join)
        return weights

    @property
    def layers(self) -> list[PhiMoEDecoderLayer]:
        return self.model.layers
