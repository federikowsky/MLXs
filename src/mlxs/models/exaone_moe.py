"""ExaOne MoE model — port from mlx_lm (exaone_moe.py), ModelProtocol-compliant.

Dense + MoE decoder with sliding-window and global attention, QK norm, RoPE,
SwitchGLU experts, optional shared experts, custom MoE gate (group_expert_select).
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
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


def _group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    """Router: sigmoid gate scores, optional group masking, top-k selection, scaling."""
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


@dataclass
class ModelArgs(BaseModelArgs):
    """ExaOne MoE config (config.json)."""

    model_type: str = "exaone_moe"
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 8192
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072
    sliding_window: int = 4096
    layer_types: list[str] | None = None
    is_moe_layer: list[bool] | None = None
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, float | str] | None = None
    rope_parameters: dict[str, Any] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.layer_types is None:
            self.layer_types = ["global"] * self.num_hidden_layers
        if self.is_moe_layer is None:
            self.is_moe_layer = [False] * self.num_hidden_layers


class MoEGate(nn.Module):
    """Router: raw weight gate + score correction bias, top-k via group_expert_select."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

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


class MLP(nn.Module):
    """Dense SwiGLU MLP (used for non-MoE layers and shared experts)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEBlock(nn.Module):
    """MoE block: MoEGate + SwitchGLU experts + optional shared expert MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
        self.gate = MoEGate(args)
        self.shared_experts: MLP | None = None
        if args.num_shared_experts and args.num_shared_experts > 0:
            self.shared_experts = MLP(
                args.hidden_size,
                args.moe_intermediate_size * args.num_shared_experts,
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class Attention(nn.Module):
    """Multi-head attention with QK norm, RoPE, sliding or global."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        layer_types = args.layer_types or []
        self.is_sliding_window = (
            layer_types[layer_idx] == "sliding_attention"
            if layer_idx < len(layer_types)
            else False
        )
        self.apply_rope_all_layers = "sliding_attention" not in layer_types
        self.use_rope = self.is_sliding_window or self.apply_rope_all_layers

        if self.use_rope:
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
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            if self.use_rope:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        elif self.use_rope:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    """Pre-norm attention + pre-norm MLP or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        is_moe = (args.is_moe_layer or [False])[layer_idx]
        self.mlp: MoEBlock | MLP = (
            MoEBlock(args) if is_moe else MLP(args.hidden_size, args.intermediate_size)
        )
        self.is_sliding_window = self.self_attn.is_sliding_window
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

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


class ExaoneMoEModel(nn.Module):
    """Transformer stack: embed, decoder layers, final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        swa_idx: int | None = None
        ga_idx: int | None = None
        for i, layer in enumerate(self.layers):
            if layer.is_sliding_window and swa_idx is None:
                swa_idx = i
            if not layer.is_sliding_window and ga_idx is None:
                ga_idx = i
        self.swa_idx = swa_idx
        self.ga_idx = ga_idx if ga_idx is not None else 0
        self.window_size = args.sliding_window

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        global_mask = create_attention_mask(
            h, cache[self.ga_idx] if self.ga_idx is not None else cache[0]
        )
        swa_mask = create_attention_mask(
            h,
            cache[self.swa_idx] if self.swa_idx is not None else cache[0],
            window_size=self.window_size,
        )

        for layer, c in zip(self.layers, cache, strict=True):
            mask = swa_mask if layer.is_sliding_window else global_mask
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """ExaOne MoE LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = ExaoneMoEModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.transformer(input_ids, cache)
        if self.args.tie_word_embeddings:
            out = self.transformer.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)  # type: ignore[assignment]
        return out

    @property
    def num_layers(self) -> int:
        return len(self.transformer.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        args = self.args
        layer_types = args.layer_types or ["global"] * args.num_hidden_layers
        return [
            (
                RotatingKVCache(max_size=args.sliding_window, keep=0)
                if (i < len(layer_types) and layer_types[i] == "sliding_attention")
                else KVCache()
            )
            for i in range(len(self.transformer.layers))
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        new_weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
        weights = new_weights

        for layer_idx in range(self.args.num_hidden_layers):
            if not (self.args.is_moe_layer or [False])[layer_idx]:
                continue
            prefix = f"model.layers.{layer_idx}"
            bias_key = f"{prefix}.mlp.e_score_correction_bias"
            if bias_key in weights:
                weights[f"{prefix}.mlp.gate.e_score_correction_bias"] = weights.pop(bias_key)
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    first_key = f"{prefix}.mlp.experts.0.{m}.{k}"
                    last_key = f"{prefix}.mlp.experts.{self.args.num_experts - 1}.{m}.{k}"
                    if first_key in weights and last_key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights
