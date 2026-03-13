"""GPT-OSS MoE model — port from mlx_lm, ModelProtocol-compliant.

Hybrid sliding-window and full attention, RoPE, SwitchGLU experts with
custom clamped SwiGLU activation. Full-attention layers use KVCache;
sliding-attention layers use RotatingKVCache. Supports attention sinks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """GPT-OSS MoE config; from_dict aligned to config.json."""

    model_type: str = "gpt_oss"
    num_hidden_layers: int = 36
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    vocab_size: int = 201088
    rms_norm_eps: float = 1e-5
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    rope_theta: float = 150000.0
    rope_scaling: dict[str, Any] | None = None
    layer_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = ["sliding_attention", "full_attention"] * (
                self.num_hidden_layers // 2
            )


def _topk(a: mx.array, k: int, axis: int = -1) -> tuple[mx.array, mx.array]:
    """Top-k values and indices along axis (MLX equivalent of torch.topk)."""
    partitioned = mx.argpartition(a, kth=-k, axis=axis)
    top_k_indices = partitioned[..., -k:]
    top_k_values = mx.take_along_axis(a, top_k_indices, axis=axis)
    return top_k_values, top_k_indices


class GptOssSwiGLU(nn.Module):
    """Clamped SwiGLU: sigmoid(alpha * clip(gate)) * gate * (linear + 1). Used only by gpt_oss."""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def __call__(self, x_linear: mx.array, x_gate: mx.array) -> mx.array:
        x_gate = mx.clip(x_gate, a_min=None, a_max=self.limit)
        x_linear = mx.clip(x_linear, a_min=-self.limit, a_max=self.limit)
        sig = mx.sigmoid(self.alpha * x_gate)
        return (x_gate * sig) * (x_linear + 1.0)


class AttentionBlock(nn.Module):
    """Multi-head attention with GQA, RoPE, and attention sinks."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.head_dim = args.head_dim
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = args.num_attention_heads // args.num_key_value_heads

        self.sinks = mx.zeros((args.num_attention_heads,))

        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * args.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * args.head_dim,
            args.hidden_size,
            bias=True,
        )

        self.sm_scale = 1.0 / math.sqrt(args.head_dim)
        self.rope = initialize_rope(
            args.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        D = self.head_dim

        q = self.q_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)
        k = self.k_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)
        v = self.v_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        v_hat = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.sm_scale, mask=mask, sinks=self.sinks
        )
        return self.o_proj(v_hat.swapaxes(1, 2).reshape(B, L, -1))


class MLPBlock(nn.Module):
    """MoE block with router, top-k expert selection, and weighted combination."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_local_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        self.experts = SwitchGLU(
            input_dims=args.hidden_size,
            hidden_dims=args.intermediate_size,
            num_experts=args.num_local_experts,
            activation=GptOssSwiGLU(),
            bias=True,
        )
        self.router = nn.Linear(args.hidden_size, args.num_local_experts, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        g = self.router(x)
        expert_logits, indices = _topk(g, k=self.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(expert_logits, axis=-1, precise=True)

        x = self.experts(x, indices)
        x = (x * mx.expand_dims(expert_weights, axis=-1)).sum(axis=-2)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm block: attention + MoE MLP with residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = AttentionBlock(args)
        self.mlp = MLPBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask, cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class GptOssMoeModel(nn.Module):
    """GPT-OSS transformer backbone: embed, alternating sliding/full layers, final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = list(args.layer_types)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.window_size = args.sliding_window
        self._swa_idx = self.layer_types.index("sliding_attention")
        self._ga_idx = self.layer_types.index("full_attention")

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        x = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        full_mask = create_attention_mask(x, cache[self._ga_idx])
        swa_mask = create_attention_mask(x, cache[self._swa_idx], window_size=self.window_size)

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types, strict=True):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            x = layer(x, mask, c)

        return self.norm(x)


class Model(nn.Module):
    """GPT-OSS LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GptOssMoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        return self.lm_head(self.model(input_ids, cache=cache))

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        caches: list[KVCache | RotatingKVCache] = []
        for lt in self.model.layer_types:
            if lt == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if any("gate_proj.weight" in k for k in weights):
            return weights

        new_weights: dict[str, Any] = {}
        for k, v in weights.items():
            if "gate_up_proj" in k and "bias" not in k:
                if "_blocks" in k:
                    v = v.astype(mx.uint32).reshape(*v.shape[:-2], -1)
                    k = k.replace("_blocks", ".weight")
                if "_scales" in k:
                    k = k.replace("_scales", ".scales")
                new_weights[k.replace("gate_up_proj", "gate_proj")] = mx.contiguous(v[..., ::2, :])
                new_weights[k.replace("gate_up_proj", "up_proj")] = mx.contiguous(v[..., 1::2, :])
            elif "down_proj" in k and "bias" not in k:
                if "_blocks" in k:
                    v = v.astype(mx.uint32).reshape(*v.shape[:-2], -1)
                    k = k.replace("_blocks", ".weight")
                if "_scales" in k:
                    k = k.replace("_scales", ".scales")
                new_weights[k] = v
            elif "gate_up_proj_bias" in k:
                new_weights[k.replace("gate_up_proj_bias", "gate_proj.bias")] = mx.contiguous(
                    v[..., ::2]
                )
                new_weights[k.replace("gate_up_proj_bias", "up_proj.bias")] = mx.contiguous(
                    v[..., 1::2]
                )
            elif "down_proj_bias" in k:
                new_weights[k.replace("down_proj_bias", "down_proj.bias")] = v
            else:
                new_weights[k] = v
        return new_weights
