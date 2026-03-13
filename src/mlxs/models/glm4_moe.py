"""GLM-4 MoE model — port from mlx_lm (glm4_moe.py), ModelProtocol-compliant.

Dense + MoE layers (first_k_dense_replace dense, rest MoE), optional QK norm,
RoPE (traditional=False), group expert selection. Optional shared experts.
Imports only from mlxs.cache, mlxs.layers, mlxs.models.base.
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
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """GLM-4 MoE config (config.json)."""

    model_type: str = "glm4_moe"
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 8192
    max_position_embeddings: int = 32768
    moe_intermediate_size: int = 4096
    norm_topk_prob: bool = True
    num_attention_heads: int = 16
    n_group: int = 1
    head_dim: int = 128
    topk_group: int = 1
    n_shared_experts: int | None = None
    n_routed_experts: int = 8
    routed_scaling_factor: float = 1.0
    num_experts_per_tok: int = 2
    first_k_dense_replace: int = 0
    num_hidden_layers: int = 28
    num_key_value_heads: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    use_qk_norm: bool = False
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    partial_rotary_factor: float = 1.0
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"


def _group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
    """Compute expert indices and scores (sigmoid, optional group masking, top-k, scaling)."""
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
        scores = scores / denominator
    scores = scores * routed_scaling_factor
    return inds, scores


class Glm4MoeAttention(nn.Module):
    """Multi-head attention with optional QK norm and partial RoPE (traditional=False)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            int(head_dim * args.partial_rotary_factor),
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
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
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Glm4MoeMLP(nn.Module):
    """Dense SwiGLU MLP (gate_proj, up_proj, down_proj)."""

    def __init__(
        self,
        args: ModelArgs,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        hidden_size = hidden_size or args.hidden_size
        intermediate_size = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Glm4MoeGate(nn.Module):
    """Router: linear gate + e_score_correction_bias, returns (inds, scores)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed_experts = args.n_routed_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,))

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


class Glm4MoeBlock(nn.Module):
    """MoE block: gate + SwitchGLU experts, optional shared experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
        )
        self.gate = Glm4MoeGate(args)
        self.shared_experts: Glm4MoeMLP | None = None
        if args.n_shared_experts is not None and args.n_shared_experts > 0:
            shared_intermediate = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = Glm4MoeMLP(args, intermediate_size=shared_intermediate)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if y.ndim == 4:
            y = y.squeeze(-2)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class Glm4MoeDecoderLayer(nn.Module):
    """Decoder: input norm → attention → residual → post_attn norm → MLP/MoE → residual."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Glm4MoeAttention(args)
        use_moe = args.n_routed_experts is not None and layer_idx >= args.first_k_dense_replace
        self.mlp = Glm4MoeBlock(args) if use_moe else Glm4MoeMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Glm4MoeModel(nn.Module):
    """GLM-4 MoE transformer: embed, decoder layers, final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Glm4MoeDecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """GLM-4 MoE LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm4MoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(self.num_layers)]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Stack expert weights from experts.{e}.{m}.{k} to switch_mlp.{m}.{k}; drop MPT layer."""
        mpt_layer = self.args.num_hidden_layers
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for _w, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    key = f"{prefix}.mlp.experts.0.{m}.{k}"
                    if key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
        return {k: v for k, v in weights.items() if not k.startswith(f"model.layers.{mpt_layer}")}
