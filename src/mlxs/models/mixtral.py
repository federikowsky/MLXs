"""Mixtral Sparse MoE model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports GQA, RoPE, and sparse mixture-of-experts with SwitchGLU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache
from mlxs.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlxs.models.rope import initialize_rope
from mlxs.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    """Mixtral model configuration."""

    model_type: str = "mixtral"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_experts_per_tok: int = 2
    num_key_value_heads: int = 8
    num_local_experts: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e6
    rope_traditional: bool = False
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
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
            queries, keys, values, cache=cache, scale=self.scale, mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class SparseMoeBlock(nn.Module):
    """Sparse mixture-of-experts block using SwitchGLU."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(self.hidden_dim, args.intermediate_size, self.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)

        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class MixtralDecoderLayer(nn.Module):
    """Pre-norm transformer block with sparse MoE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.block_sparse_moe = SparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.block_sparse_moe(self.post_attention_layernorm(h))


class MixtralModel(nn.Module):
    """Mixtral transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [MixtralDecoderLayer(args) for _ in range(args.num_hidden_layers)]
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
    """Mixtral LM head wrapper -- satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MixtralModel(args)
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
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.block_sparse_moe.experts.0.{n}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.block_sparse_moe.experts.{e}.{n}.{k}")
                            for e in range(self.args.num_local_experts)
                        ]
                        weights[f"{prefix}.block_sparse_moe.switch_mlp.{m}.{k}"] = (
                            mx.stack(to_join)
                        )
        return weights

    @property
    def layers(self) -> list[MixtralDecoderLayer]:
        return self.model.layers
