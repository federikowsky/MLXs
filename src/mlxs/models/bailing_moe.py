"""Bailing MoE model: sparse MoE with optional shared experts and group top-k routing.

Implements ModelProtocol. Uses SwitchGLU from mlxs.layers.moe, RoPE, QK-norm option.
Compatible with mlx_lm-converted weights.
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
    """Bailing MoE model configuration."""

    model_type: str = "bailing_moe"
    hidden_size: int = 2048
    intermediate_size: int = 5632
    max_position_embeddings: int = 32768
    moe_intermediate_size: int = 1408
    num_experts: int = 8
    num_shared_experts: int = 0
    norm_topk_prob: bool = True
    num_attention_heads: int = 16
    num_experts_per_tok: int = 2
    num_hidden_layers: int = 24
    num_key_value_heads: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    vocab_size: int = 128256
    first_k_dense_replace: int = 0
    rope_scaling: dict[str, Any] | None = None
    use_bias: bool = False
    use_qkv_bias: bool = False
    norm_head: bool = False
    norm_softmax: bool = False
    use_qk_norm: bool = False
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0
    rotary_dim: int | None = None
    moe_router_enable_expert_bias: bool = False
    moe_router_enable_routed_scaling: bool = True
    routed_scaling_factor: float = 1.0
    score_function: str = "softmax"
    n_group: int = 1
    topk_group: int = 4
    moe_shared_expert_intermediate_size: int | None = None
    moe_router_enable_shared_expert: bool = True


def _aggregate_expert_outputs(expert_outputs: mx.array, scores: mx.array) -> mx.array:
    return (expert_outputs * scores[..., None]).sum(axis=-2).astype(expert_outputs.dtype)


def _group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array | None,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    score_function: str,
) -> tuple[mx.array, mx.array]:
    in_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    orig_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator
    scores = scores * routed_scaling_factor
    return inds, scores.astype(in_type)


class BailingMoeMLP(nn.Module):
    """Dense SwiGLU MLP used for dense layers or shared experts."""

    def __init__(self, args: ModelArgs, intermediate_size: int | None = None) -> None:
        super().__init__()
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else args.intermediate_size
        )
        self.gate_proj = nn.Linear(args.hidden_size, self.intermediate_size, bias=args.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, args.hidden_size, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, self.intermediate_size, bias=args.use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class BailingMoeAttention(nn.Module):
    """Multi-head attention with optional QK-norm and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.use_qk_norm = args.use_qk_norm
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope_dim = args.rotary_dim or int(self.head_dim * args.partial_rotary_factor)
        self.rope = initialize_rope(
            rope_dim,
            args.rope_theta,
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

        qkv = self.query_key_value(x)
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        queries = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

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
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class BailingMoeGate(nn.Module):
    """Router with group top-k and score normalization."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.enable_routed_scaling = args.moe_router_enable_routed_scaling

        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,)) if args.moe_router_enable_expert_bias else None
        )
        self.score_function = args.score_function

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        scale = self.routed_scaling_factor if self.enable_routed_scaling else 1.0
        return _group_expert_select(
            self.gate_proj(x),
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            scale,
            self.norm_topk_prob,
            self.score_function,
        )


class BailingMoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with SwitchGLU and optional shared experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = BailingMoeGate(args)
        shared_dim = args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        self.shared_experts = (
            BailingMoeMLP(
                args=args,
                intermediate_size=shared_dim * args.num_shared_experts,
            )
            if args.num_shared_experts > 0 and args.moe_router_enable_shared_expert
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = _aggregate_expert_outputs(out, topk_weight)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class BailingMoeDecoderLayer(nn.Module):
    """Pre-norm decoder layer: attention + dense or sparse MoE MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.attention = BailingMoeAttention(args)
        self.mlp = (
            BailingMoeSparseMoeBlock(args)
            if (args.num_experts is not None and layer_idx >= args.first_k_dense_replace)
            else BailingMoeMLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class BailingMoeModel(nn.Module):
    """Bailing MoE transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BailingMoeDecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0] if cache else None)
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    """Bailing MoE LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.norm_head = args.norm_head
        self.model_type = args.model_type
        self.model = BailingMoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

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
        if self.norm_head and "lm_head.weight" in weights:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            weight_norm = mx.linalg.norm(w.astype(mx.float32), axis=0, keepdims=True) + 1e-7
            weights["lm_head.weight"] = (w / weight_norm).astype(dtype)

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            if layer_idx >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        key = f"{prefix}.mlp.experts.0.{m}.{k}"
                        if key in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
                gate_weight_key = f"{prefix}.mlp.gate.weight"
                if gate_weight_key in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = weights.pop(gate_weight_key)
                gate_bias_key = f"{prefix}.mlp.gate.bias"
                if gate_bias_key in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.bias"] = weights.pop(gate_bias_key)
        return weights
