"""Step3p5 model: hybrid full/sliding attention, MoE with SwitchGLU, optional clamped SwiGLU.

Implements ModelProtocol. Port from mlx_lm. Uses nn.RMSNorm (shared norm), RoPE
(optional scaling), KVCache / RotatingKVCache, create_attention_mask. Imports:
mlxs.cache, mlxs.layers, mlxs.models.base.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache import KVCache, RotatingKVCache
from mlxs.cache.attention_mask import create_attention_mask
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwiGLU, SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@partial(mx.compile, shapeless=True)
def _clamped_swiglu(x: mx.array, gate: mx.array, limit: float) -> mx.array:
    """Clamped SwiGLU for Step3p5 MLP/MoE (stability)."""
    gate = mx.clip(nn.silu(gate), a_min=None, a_max=limit)
    x = mx.clip(x, a_min=-limit, a_max=limit)
    return gate * x


class _ClampedSwiGLU(nn.Module):
    """SwiGLU with optional clamp on gate and up (Step3p5-specific)."""

    def __init__(self, limit: float) -> None:
        super().__init__()
        self.limit = limit

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _clamped_swiglu(x, gate, self.limit)


@dataclass
class ModelArgs(BaseModelArgs):
    """Step3p5 model configuration."""

    model_type: str = "step3p5"
    hidden_size: int = 2048
    num_hidden_layers: int = 32
    vocab_size: int = 128256
    num_attention_heads: int = 16
    num_attention_groups: int = 16
    head_dim: int = 128
    intermediate_size: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float | list[float] = 10000.0
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 262144
    sliding_window: int = 512
    layer_types: list[str] | None = None
    yarn_only_types: list[str] | None = None
    partial_rotary_factors: list[float] | None = None
    attention_other_setting: dict[str, Any] | None = None
    use_head_wise_attn_gate: bool = True
    moe_num_experts: int = 288
    moe_top_k: int = 8
    moe_intermediate_size: int = 1280
    share_expert_dim: int = 1280
    moe_layers_enum: str | None = None
    moe_router_scaling_factor: float = 3.0
    norm_expert_weight: bool = True
    swiglu_limits: list[float] | None = None
    swiglu_limits_shared: list[float] | None = None
    tie_word_embeddings: bool = False


@mx.compile
def _moe_gate_select(
    gates: mx.array,
    router_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    """Top-k gate selection with optional score correction and normalization."""
    scores = mx.sigmoid(gates.astype(mx.float32))
    corrected_scores = scores + router_bias
    topk_indices = mx.argpartition(-corrected_scores, kth=top_k - 1, axis=-1)[..., :top_k]
    topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)
    if norm_topk_prob:
        topk_weights = topk_weights / (mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20)
    return topk_indices, topk_weights * routed_scaling_factor


class _Step3p5MoEGate(nn.Module):
    """Router: gate linear + top-k selection with optional norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.top_k = args.moe_top_k
        self.routed_scaling_factor = args.moe_router_scaling_factor
        self.norm_topk_prob = args.norm_expert_weight
        self.gate = nn.Linear(args.hidden_size, args.moe_num_experts, bias=False)
        self.router_bias = mx.zeros((args.moe_num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return _moe_gate_select(
            self.gate(x),
            self.router_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class _Step3p5MLP(nn.Module):
    """Dense MLP with optional clamped SwiGLU."""

    def __init__(
        self,
        args: ModelArgs,
        intermediate_size: int,
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.limit = swiglu_limit if swiglu_limit and swiglu_limit > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.limit is not None:
            return self.down_proj(_clamped_swiglu(up, gate, self.limit))
        return self.down_proj(swiglu(gate, up))


class _Step3p5MoE(nn.Module):
    """MoE block: gate + SwitchGLU experts (optional ClampedSwiGLU) + shared expert MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        swiglu_limit = 0.0
        if args.swiglu_limits and layer_idx < len(args.swiglu_limits):
            swiglu_limit = args.swiglu_limits[layer_idx] or 0.0
        swiglu_limit_shared = 0.0
        if args.swiglu_limits_shared and layer_idx < len(args.swiglu_limits_shared):
            swiglu_limit_shared = args.swiglu_limits_shared[layer_idx] or 0.0

        activation: nn.Module = _ClampedSwiGLU(swiglu_limit) if swiglu_limit > 0 else SwiGLU()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.moe_num_experts,
            activation=activation,
        )
        self.share_expert = _Step3p5MLP(
            args,
            intermediate_size=args.share_expert_dim,
            swiglu_limit=swiglu_limit_shared,
        )
        self.gate = _Step3p5MoEGate(args)

    def __call__(self, x: mx.array) -> mx.array:
        topk_indices, topk_weights = self.gate(x)
        routed = self.switch_mlp(x, topk_indices)
        routed = (routed * topk_weights[..., None]).sum(axis=-2).astype(routed.dtype)
        return routed + self.share_expert(x)


def _moe_layers_set(args: ModelArgs) -> set[int]:
    """Set of layer indices that are MoE layers."""
    if args.moe_layers_enum:
        return {int(i) for i in args.moe_layers_enum.strip().split(",")}
    return set(range(1, args.num_hidden_layers))


class _Step3p5Attention(nn.Module):
    """Multi-head attention with Q/K RMSNorm, RoPE, optional head-wise gate."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        layer_types = args.layer_types or []
        self.is_sliding = (
            layer_types[layer_idx] == "sliding_attention" if layer_types else layer_idx % 2 == 0
        )
        if self.is_sliding and args.attention_other_setting:
            self.num_heads = args.attention_other_setting["num_attention_heads"]
            self.num_kv_heads = args.attention_other_setting["num_attention_groups"]
        else:
            self.num_heads = args.num_attention_heads
            self.num_kv_heads = args.num_attention_groups
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.use_head_wise_attn_gate = args.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(dim, self.num_heads, bias=False)

        rope_theta: float = (
            args.rope_theta[layer_idx]
            if isinstance(args.rope_theta, list) and layer_idx < len(args.rope_theta)
            else (args.rope_theta if isinstance(args.rope_theta, (int, float)) else 10000.0)
        )
        partial_rotary_factor = 1.0
        if args.partial_rotary_factors and layer_idx < len(args.partial_rotary_factors):
            partial_rotary_factor = args.partial_rotary_factors[layer_idx]
        rope_dims = int(self.head_dim * partial_rotary_factor)

        yarn_only_types = args.yarn_only_types or []
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        rope_scaling = None
        if not (yarn_only_types and layer_type not in yarn_only_types):
            rope_scaling = args.rope_scaling

        self.rope = initialize_rope(
            dims=rope_dims,
            base=rope_theta,
            traditional=False,
            scaling_config=rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.num_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
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
        output = output.transpose(0, 2, 1, 3)
        if self.use_head_wise_attn_gate:
            output = output * mx.sigmoid(self.g_proj(x))[..., None]
        return self.o_proj(output.reshape(B, L, -1))


class _Step3p5DecoderLayer(nn.Module):
    """Pre-norm decoder layer: attention + MLP or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = _Step3p5Attention(args, layer_idx)
        self.is_sliding = self.self_attn.is_sliding
        moe_layers_idx = _moe_layers_set(args)
        self.is_moe_layer = layer_idx in moe_layers_idx
        if self.is_moe_layer:
            self.mlp = _Step3p5MoE(args, layer_idx)
        else:
            swiglu_limit = 0.0
            if args.swiglu_limits_shared and layer_idx < len(args.swiglu_limits_shared):
                swiglu_limit = args.swiglu_limits_shared[layer_idx] or 0.0
            self.mlp = _Step3p5MLP(
                args,
                intermediate_size=args.intermediate_size,
                swiglu_limit=swiglu_limit,
            )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class _Step3p5Model(nn.Module):
    """Backbone: embed + decoder layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [_Step3p5DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._swa_idx = next((i for i, layer in enumerate(self.layers) if layer.is_sliding), None)
        self._full_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_sliding), None
        )

    def __call__(
        self,
        x: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = None
        swa_mask = None
        if self._full_idx is not None:
            full_mask = create_attention_mask(h, cache[self._full_idx])
        if self._swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self._swa_idx], window_size=self.args.sliding_window
            )
        for layer, c in zip(self.layers, cache, strict=True):
            mask = swa_mask if layer.is_sliding else full_mask
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """Step3p5 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = _Step3p5Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        h = self.model(input_ids, cache=cache)
        return self.lm_head(h)

    @property
    def num_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return [
            RotatingKVCache(max_size=self.args.sliding_window) if layer.is_sliding else KVCache()
            for layer in self.model.layers
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Remap MoE keys and optional +1 on norm weights for vanilla checkpoints."""
        remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ]
        is_vanilla = any(src in k and dst not in k for k in weights for src, dst in remappings)
        new_weights: dict[str, Any] = {}
        for k, v in weights.items():
            if ".mtp" in k:
                continue
            if "model.layers." in k:
                parts = k.split(".")
                if (
                    len(parts) > 2
                    and parts[2].isdigit()
                    and int(parts[2]) >= self.args.num_hidden_layers
                ):
                    continue
            for src, dst in remappings:
                if src in k and dst not in k:
                    k = k.replace(src, dst)
                    break
            if is_vanilla and k.endswith(".weight") and "norm" in k:
                v = v + 1
            new_weights[k] = v
        return new_weights
