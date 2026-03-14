"""Qwen3 MoE model architecture.

Implements ModelProtocol. Supports sparse MoE with QK-norm attention
and configurable dense/MoE layer placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen3 MoE model configuration."""

    model_type: str = "qwen3_moe"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    num_experts: int = 64
    num_experts_per_tok: int = 8
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)
    moe_intermediate_size: int = 1024
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 32768
    norm_topk_prob: bool = True
    rope_scaling: dict[str, float | str] | None = None


class Attention(nn.Module):
    """Multi-head attention with GQA, QK-norm, and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
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
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with top-k routing."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, self.num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        return y


class Qwen3MoeDecoderLayer(nn.Module):
    """Transformer block with configurable dense or MoE MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        is_moe = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.mlp: Qwen3MoeSparseMoeBlock | MLP = (
            Qwen3MoeSparseMoeBlock(args) if is_moe
            else MLP(args.hidden_size, args.intermediate_size)
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3MoeModel(nn.Module):
    """Qwen3 MoE transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3MoeDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    """Qwen3 MoE LM head wrapper -- satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3MoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
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
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.mlp.experts.0.{n}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
        return weights

    @property
    def layers(self) -> list[Qwen3MoeDecoderLayer]:
        return self.model.layers
