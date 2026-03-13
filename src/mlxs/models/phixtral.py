"""Phixtral: Phi-style decoder with MoE (parallel attention + MoE, LayerNorm).

Ported from mlx_lm. Implements ModelProtocol. Uses LayerNorm, RoPE (partial),
single QKV projection, and SwitchMLP (GELU, bias) for experts.
Imports: mlxs.cache, mlxs.layers, mlxs.models.base.
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
from mlxs.layers.moe import SwitchMLP
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Phixtral model configuration (mlx_lm-compatible)."""

    model_type: str = "phixtral"
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32
    num_experts_per_tok: int = 2
    num_local_experts: int = 4
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class RoPEAttention(nn.Module):
    """Multi-head attention with partial RoPE and single QKV projection."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        rotary_dim: int,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.rope = nn.RoPE(rotary_dim, traditional=False, base=rope_theta)
        self.wqkv = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        qkv = self.wqkv(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)
        B, L, _ = queries.shape
        num_heads = self.num_heads
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        scale = math.sqrt(1 / queries.shape[-1])
        queries = queries.astype(mx.float32)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        ).astype(values.dtype)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class PhixtralMoE(nn.Module):
    """MoE block with gate and SwitchMLP (GELU, bias)."""

    def __init__(self, args: ModelArgs, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.gate = nn.Linear(args.model_dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchMLP(
            dim, hidden_dim, self.num_experts, activation=nn.GELU(approx="precise"), bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class ParallelBlock(nn.Module):
    """Parallel attention + MoE with LayerNorm (Phi-style)."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        dims = config.model_dim
        mlp_dims = dims * 4
        self.mixer = RoPEAttention(
            dims, config.num_heads, config.rotary_dim, config.rope_theta
        )
        self.ln = nn.LayerNorm(dims, eps=config.layer_norm_eps)
        self.moe = PhixtralMoE(config, dims, mlp_dims)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None,
    ) -> mx.array:
        h = self.ln(x)
        attn_h = self.mixer(h, mask, cache)
        ff_h = self.moe(h)
        return attn_h + ff_h + x


class PhixtralTransformer(nn.Module):
    """Phixtral backbone: embed + stacked parallel blocks."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.embd = nn.Embedding(config.num_vocab, config.model_dim)
        self.layers = [ParallelBlock(config) for _ in range(config.num_layers)]

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: list[KVCache] | None,
    ) -> mx.array:
        x = self.embd(x)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        for layer, c in zip(self.layers, cache, strict=True):
            x = layer(x, mask, c)
        return x


class OutputHead(nn.Module):
    """Final LayerNorm + linear to vocab."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)
        self.linear = nn.Linear(config.model_dim, config.num_vocab)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.linear(self.ln(inputs))


class Model(nn.Module):
    """Phixtral LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = PhixtralTransformer(args)
        self.lm_head = OutputHead(args)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | str | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        if mask is None:
            h = self.transformer.embd(input_ids)
            mask = create_attention_mask(h, cache[0] if cache else None)
        y = self.transformer(input_ids, mask, cache)
        return self.lm_head(y)

    @property
    def num_layers(self) -> int:
        return len(self.transformer.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.num_vocab

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.transformer.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if "transformer.h.0.moe.mlp.0.fc1.weight" not in weights:
            return weights
        for layer_idx in range(self.args.num_layers):
            prefix = f"transformer.h.{layer_idx}"
            for n in ["fc1", "fc2"]:
                for k in ["weight", "scales", "biases", "bias"]:
                    key0 = f"{prefix}.moe.mlp.0.{n}.{k}"
                    if key0 not in weights:
                        continue
                    expert_keys = [
                        weights.pop(f"{prefix}.moe.mlp.{e}.{n}.{k}")
                        for e in range(self.args.num_local_experts)
                    ]
                    weights[f"{prefix}.moe.switch_mlp.{n}.{k}"] = mx.stack(expert_keys)
        return weights
