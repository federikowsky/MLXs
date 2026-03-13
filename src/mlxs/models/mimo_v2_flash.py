"""Mimo V2 Flash model — hybrid full/sliding attention + MoE, ModelProtocol-compliant.

Ported from mlx_lm models/mimo_v2_flash. Full vs sliding-window attention per
hybrid_layer_pattern; MoE (group expert select + SwitchGLU) per moe_layer_freq.
Uses KVCache for full-attn layers, RotatingKVCache for sliding. RMSNorm only.
Imports: mlxs.cache, mlxs.layers, mlxs.models.base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Mimo V2 Flash config; from_dict aligned to config.json."""

    model_type: str = "mimo_v2_flash"
    num_experts_per_tok: int = 1
    hybrid_layer_pattern: list[int] = ()
    moe_layer_freq: list[int] = ()
    add_swa_attention_sink_bias: bool = False
    add_full_attention_sink_bias: bool = False
    sliding_window_size: int = 4096
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 5504
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    n_shared_experts: int | None = None
    n_routed_experts: int | None = 8
    routed_scaling_factor: float | None = 1.0
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = False
    n_group: int = 1
    topk_group: int = 1
    max_position_embeddings: int = 32768
    layernorm_epsilon: float = 1e-6
    rope_theta: float = 10000.0
    swa_rope_theta: float = 10000.0
    swa_num_attention_heads: int = 16
    swa_num_key_value_heads: int = 16
    head_dim: int = 128
    v_head_dim: int = 128
    swa_head_dim: int = 128
    swa_v_head_dim: int = 128
    partial_rotary_factor: int = 1


def _group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    """Grouped expert selection: sigmoid gates + group top-k, return indices and scores."""
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
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
        scores = scores / (denominator + 1e-20)
    scores = scores * routed_scaling_factor
    return inds, scores


class Attention(nn.Module):
    """Full or sliding-window attention with optional sink bias; RoPE on partial dims."""

    def __init__(self, args: ModelArgs, is_sliding_window: bool) -> None:
        super().__init__()
        dim = args.hidden_size
        self.is_sliding_window = is_sliding_window
        if is_sliding_window:
            n_heads = args.swa_num_attention_heads
            n_kv_heads = args.swa_num_key_value_heads
            self.has_sinks = args.add_swa_attention_sink_bias
            head_dim = args.swa_head_dim
            v_head_dim = args.swa_v_head_dim
            rope_theta = args.swa_rope_theta
        else:
            n_heads = args.num_attention_heads
            n_kv_heads = args.num_key_value_heads
            self.has_sinks = args.add_full_attention_sink_bias
            head_dim = args.head_dim
            v_head_dim = args.v_head_dim
            rope_theta = args.rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.scale = head_dim**-0.5
        rope_dim = int(args.partial_rotary_factor * head_dim)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * v_head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * v_head_dim, dim, bias=False)
        self.rope = nn.RoPE(rope_dim, traditional=False, base=rope_theta)
        if self.has_sinks:
            self.attention_sink_bias = mx.ones((n_heads,))
        else:
            self.attention_sink_bias = None

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

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        out = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache if cache is not None else None,
            scale=self.scale,
            mask=mask,
            sinks=self.attention_sink_bias,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU MLP (gate + up -> swiglu -> down)."""

    def __init__(
        self,
        args: ModelArgs,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            args.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    """Router with group expert selection (noaux_tc)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.topk_method == "noaux_tc", "Unsupported topk method."
        n_routed = args.n_routed_experts or 8
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed_experts = n_routed
        self.routed_scaling_factor = args.routed_scaling_factor or 1.0
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = mx.zeros((n_routed, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((n_routed,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        gates = x @ self.weight.T
        return _group_expert_select(
            gates,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    """MoE block: MoEGate + SwitchGLU experts, optional shared experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        n_routed = args.n_routed_experts or 8
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            n_routed,
        )
        self.gate = MoEGate(args)
        if args.n_shared_experts is not None:
            inter = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = MLP(args, intermediate_size=inter)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    """Pre-norm: input_layernorm -> attn -> residual; post_attn_layernorm -> mlp -> residual."""

    def __init__(self, args: ModelArgs, is_moe: bool, is_sliding_window: bool) -> None:
        super().__init__()
        self.self_attn = Attention(args, is_sliding_window)
        self.mlp = MoE(args) if is_moe else MLP(args)
        self.is_sliding_window = is_sliding_window
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(nn.Module):
    """Embedding + decoder layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args,
                is_moe=(args.moe_layer_freq[idx] == 1),
                is_sliding_window=(args.hybrid_layer_pattern[idx] == 1),
            )
            for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.sliding_window_size = args.sliding_window_size

    def __call__(
        self,
        x: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        for layer, c in zip(self.layers, cache, strict=True):
            mask = create_attention_mask(
                h, c, window_size=self.sliding_window_size if layer.is_sliding_window else None
            )
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """Mimo V2 Flash: LanguageModel + lm_head; ModelProtocol-compliant."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        return self.lm_head(out)

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        caches: list[KVCache | RotatingKVCache] = []
        for layer in self.model.layers:
            if layer.is_sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window_size, keep=0))
            else:
                caches.append(KVCache())
        return caches

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Dequantize fp8 weights, stack expert weights, remove mtp keys."""

        def dequant(weight: mx.array, scale_inv: mx.array) -> mx.array:
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128
            m, n = weight.shape
            pad_bottom = bs * scale_inv.shape[0] - m
            pad_side = bs * scale_inv.shape[1] - n
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs))
            weight = (weight * scale_inv[:, None, :, None]).reshape(m + pad_bottom, n + pad_side)
            return weight[:m, :n].astype(mx.bfloat16)

        new_weights: dict[str, Any] = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                w = weights[wk]
                new_weights[wk] = dequant(w, scale_inv)
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        n_routed = self.args.n_routed_experts or 8
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for _, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for key in ["weight", "scales", "biases"]:
                    name = f"{prefix}.mlp.experts.0.{m}.{key}"
                    if name in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{key}")
                            for e in range(n_routed)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{key}"] = mx.stack(to_join)

        return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}
