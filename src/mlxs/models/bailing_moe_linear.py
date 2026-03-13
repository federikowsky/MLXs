"""Bailing MoE Linear: hybrid GLA (linear) + attention layers with sparse MoE.

Implements ModelProtocol. Alternating linear attention (recurrent GLA) and
full attention by layer_group_size; MoE blocks use SwitchGLU + optional shared expert.
Imports: mlxs.cache, mlxs.layers, mlxs.models.base. GroupRMSNorm is model-specific.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Bailing MoE Linear config (from config.json / mlx_lm)."""

    model_type: str = "bailing_moe_linear"
    hidden_size: int = 2048
    intermediate_size: int = 8192
    max_position_embeddings: int = 32768
    moe_intermediate_size: int = 1408
    num_experts: int = 64
    num_shared_experts: int = 2
    norm_topk_prob: bool = True
    num_attention_heads: int = 16
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 32
    num_key_value_heads: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    vocab_size: int = 128256
    first_k_dense_replace: int = 0
    layer_group_size: int = 4
    group_norm_size: int = 4
    rope_scaling: dict[str, Any] | None = None
    rope_traditional: bool = False
    use_bias: bool = False
    use_qkv_bias: bool = False
    norm_head: bool = False
    norm_softmax: bool = False
    use_qk_norm: bool = False
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0
    moe_router_enable_expert_bias: bool = False
    moe_router_enable_routed_scaling: bool = True
    routed_scaling_factor: float = 1.0
    score_function: str = "softmax"
    n_group: int = 1
    topk_group: int = 4
    use_rmsnorm: bool = True
    moe_shared_expert_intermediate_size: int | None = None
    moe_router_enable_shared_expert: bool = True
    head_dim: int | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        sig = inspect.signature(cls)
        return cls(**{k: v for k, v in params.items() if k in sig.parameters})


def _recurrent_gla(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    scale: float,
    h: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Recurrence per (b, h): h_t = h_{t-1}*exp(g_t) + k_t^T@v_t, y_t = (q_t @ h_t)*scale."""
    L = q.shape[2]
    exp_g = mx.exp(g)[:, None, None].astype(q.dtype)
    q = q * scale
    outputs = []
    for t in range(L):
        q_t = q[:, :, t : t + 1]
        k_t = k[:, :, t : t + 1]
        v_t = v[:, :, t : t + 1]
        h_up = k_t.transpose(0, 1, 3, 2) @ v_t
        h = h * exp_g + h_up if h is not None else h_up
        o_t = q_t @ h
        outputs.append(o_t)
    return mx.concatenate(outputs, axis=2), h


class GroupRMSNorm(nn.Module):
    """RMSNorm over groups (model-specific; used only by bailing_moe_linear)."""

    def __init__(self, dims: int, eps: float = 1e-5, groups: int = 1) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.unflatten(x, axis=-1, shape=(self.groups, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * mx.flatten(x, -2)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, args: ModelArgs, intermediate_size: int | None = None) -> None:
        super().__init__()
        mid = intermediate_size if intermediate_size is not None else args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, mid, bias=args.use_bias)
        self.down_proj = nn.Linear(mid, args.hidden_size, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, mid, bias=args.use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    """Full attention with RoPE, optional QK-norm, SDPA."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.use_qk_norm = args.use_qk_norm
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (args.hidden_size // self.num_attention_heads)
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
        else:
            self.key_layernorm = None
            self.query_layernorm = None

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            args.rope_theta,
            traditional=args.rope_traditional,
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

        if self.query_layernorm is not None:
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


def _gla_slopes(num_heads: int, num_layers: int, layer_idx: int) -> mx.array:
    def power_of_2_slopes(n: int) -> list[float]:
        return [2 ** (-(2 ** -(math.log2(n) - 3)) * (i + 1)) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = power_of_2_slopes(num_heads)
    else:
        p = 2 ** math.floor(math.log2(num_heads))
        slopes = power_of_2_slopes(p) + power_of_2_slopes(2 * p)[::2][: num_heads - p]
    slopes = mx.array(slopes, dtype=mx.float32)
    denom = max(1, num_layers - 1)
    layer_pos = max(0, layer_idx - 1)
    layer_factor = 1 - (layer_pos / denom) + 1e-5
    return -slopes * layer_factor


class LinearAttention(nn.Module):
    """GLA (recurrent linear attention) with RoPE and optional QK-norm."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.use_qk_norm = args.use_qk_norm
        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_attention_heads
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
        self.g_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.g_norm = GroupRMSNorm(
            args.num_attention_heads * self.head_dim,
            eps=args.rms_norm_eps,
            groups=args.group_norm_size,
        )
        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        else:
            self.key_layernorm = None
            self.query_layernorm = None

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        self._slope = _gla_slopes(self.num_attention_heads, args.num_hidden_layers, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
        offset: int = 0,
    ) -> mx.array:
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv_mix = qkv.reshape(
            B,
            L,
            (self.num_attention_heads + 2 * self.num_key_value_heads),
            self.head_dim,
        )
        q, k, v = mx.split(
            qkv_mix,
            [
                self.num_attention_heads,
                self.num_attention_heads + self.num_key_value_heads,
            ],
            axis=2,
        )
        queries = q.transpose(0, 2, 1, 3)
        keys = k.transpose(0, 2, 1, 3)
        values = v.transpose(0, 2, 1, 3)

        if self.query_layernorm is not None:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        h_state = cache[0] if cache is not None else None
        output, new_h = _recurrent_gla(
            q=queries, k=keys, v=values, g=self._slope, scale=self.scale, h=h_state
        )
        if cache is not None:
            cache[0] = new_h
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.g_norm(output) * mx.sigmoid(self.g_proj(x))
        return self.dense(output)


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
        scores = mx.put_along_axis(scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor
    return inds, scores.astype(in_type)


class Gate(nn.Module):
    """MoE router with group expert selection."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function
        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,)) if args.moe_router_enable_expert_bias else None
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return _group_expert_select(
            self.gate_proj(x),
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class SparseMoeBlock(nn.Module):
    """Sparse MoE: SwitchGLU experts + optional shared expert."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = Gate(args)
        shared_dim = args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        self.shared_experts = (
            MLP(args, intermediate_size=shared_dim * args.num_shared_experts)
            if args.num_shared_experts > 0 and args.moe_router_enable_shared_expert
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = (out * topk_weight[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class DecoderLayer(nn.Module):
    """One block: attention (global or linear) + MLP or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.is_global = (
            (layer_idx + 1) % args.layer_group_size == 0
            or layer_idx >= args.num_hidden_layers // args.layer_group_size * args.layer_group_size
        )
        if self.is_global:
            self.attention = Attention(args)
        else:
            self.attention = LinearAttention(args, layer_idx=layer_idx)

        self.mlp = (
            SparseMoeBlock(args)
            if (args.num_experts is not None and layer_idx >= args.first_k_dense_replace)
            else MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ArraysCache | None = None,
        offset: int = 0,
    ) -> mx.array:
        if self.is_global:
            r = self.attention(self.input_layernorm(x), mask, cache)
        else:
            r = self.attention(self.input_layernorm(x), mask, cache, offset=offset)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(nn.Module):
    """Embed + decoder layers + final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.gla_idx = 0
        self.attn_idx = args.layer_group_size - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        offset = 0
        attn_mask = create_attention_mask(h, cache[self.attn_idx])
        gla_mask = create_ssm_mask(h, cache[self.gla_idx])
        if cache[self.attn_idx] is not None:
            offset = cache[self.attn_idx].offset

        for layer, c in zip(self.layers, cache, strict=True):
            mask = attn_mask if layer.is_global else gla_mask
            h = layer(h, mask, c, offset=offset)

        return self.norm(h)


class Model(nn.Module):
    """Bailing MoE Linear: satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.norm_head = args.norm_head
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | ArraysCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def num_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return [
            KVCache() if layer.is_global else ArraysCache(size=1) for layer in self.model.layers
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        if self.norm_head:
            w = weights.get("lm_head.weight")
            if w is not None:
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
                gate_w = f"{prefix}.mlp.gate.weight"
                if gate_w in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = weights.pop(gate_w)
                gate_b = f"{prefix}.mlp.gate.bias"
                if gate_b in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.bias"] = weights.pop(gate_b)
        return weights
