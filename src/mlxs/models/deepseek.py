"""DeepSeek (v1) model architecture with sparse MoE.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports GQA, RoPE (with linear scaling), SwiGLU, and sparse MoE
with shared experts and SwitchGLU routing.
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
    """DeepSeek v1 model configuration."""

    model_type: str = "deepseek"
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: int | None = None
    n_routed_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        bias = args.attention_bias

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=bias)

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


class DeepseekMLP(nn.Module):
    """SwiGLU MLP with configurable hidden/intermediate sizes."""

    def __init__(
        self,
        args: ModelArgs,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size or args.hidden_size
        self.intermediate_size = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    """Top-k expert gating for DeepSeek MoE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.weight = mx.zeros((self.n_routed_experts, args.hidden_size))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        gates = x @ self.weight.T
        scores = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(scores, inds, axis=-1)
        return inds, scores


class DeepseekMoE(nn.Module):
    """Sparse MoE block with optional shared experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self._n_shared_experts = args.n_shared_experts
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
        )
        self.gate = MoEGate(args)
        if args.n_shared_experts is not None:
            intermediate_size = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = DeepseekMLP(
                args, intermediate_size=intermediate_size,
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self._n_shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class DeepseekDecoderLayer(nn.Module):
    """Pre-norm transformer block with dense MLP or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp: DeepseekMoE | DeepseekMLP = (
            DeepseekMoE(args)
            if (
                args.n_routed_experts is not None
                and layer_idx >= args.first_k_dense_replace
                and layer_idx % args.moe_layer_freq == 0
            )
            else DeepseekMLP(args)
        )
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


class DeepseekModel(nn.Module):
    """DeepSeek transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DeepseekDecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
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
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """DeepSeek LM head wrapper -- satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

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
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
        return weights

    @property
    def layers(self) -> list[DeepseekDecoderLayer]:
        return self.model.layers
