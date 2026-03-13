"""LFM2 MoE hybrid model (short conv + full attention + dense/MoE FFN).

Port from mlx_lm. Implements ModelProtocol. Alternating short-conv and
full-attention layers; dense MLP for first num_dense_layers, then
SwitchGLU MoE. Uses KVCache for attention and ArraysCache(size=1) for conv.
"""

from __future__ import annotations

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
    """LFM2 MoE config; from_dict maps config.json (same field names as mlx_lm/HF)."""

    model_type: str = "lfm2_moe"
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 24
    num_experts: int = 8
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 131072
    use_expert_bias: bool = False
    num_dense_layers: int = 2
    norm_eps: float = 1e-6
    conv_bias: bool = False
    conv_L_cache: int = 4
    rope_theta: float = 1000000.0
    rope_parameters: dict[str, Any] | None = None
    full_attn_idxs: list[int] | None = None
    layer_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = float(self.rope_parameters["rope_theta"])
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.full_attn_idxs is None and self.layer_types is not None:
            self.full_attn_idxs = [
                i for i, lt in enumerate(self.layer_types) if lt == "full_attention"
            ]
        if self.full_attn_idxs is None:
            self.full_attn_idxs = []


class Attention(nn.Module):
    """Multi-head attention with per-head Q/K RMSNorm and RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
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

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(queries, keys, values, cache, self.scale, mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    """Gated depthwise conv block with recurrent state in ArraysCache."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.L_cache = args.conv_L_cache
        self.conv = nn.Conv1d(
            args.hidden_size,
            args.hidden_size,
            kernel_size=self.L_cache,
            groups=args.hidden_size,
            bias=args.conv_bias,
        )
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=args.conv_bias)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        B, T, _ = x.shape
        g1, g2, inp = mx.split(self.in_proj(x), 3, axis=-1)
        gated = g1 * inp
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0.0)

        if cache is not None:
            state = cache[0]
            if state is None:
                state = mx.zeros(
                    (B, self.L_cache - 1, self.args.hidden_size),
                    dtype=gated.dtype,
                )
            gated = mx.concatenate([state, gated], axis=1)
            n_keep = self.L_cache - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, T)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(gated, positions, axis=1)
            else:
                cache[0] = gated[:, -n_keep:, :]
            cache.advance(T)
        else:
            gated = mx.pad(gated, [(0, 0), (self.L_cache - 1, 0), (0, 0)])

        conv_out = self.conv(gated)
        return self.out_proj(g2 * conv_out)


class MLP(nn.Module):
    """SwiGLU MLP (dense layers)."""

    def __init__(self, args: ModelArgs, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Lfm2MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block: gate + top-k routing + SwitchGLU experts and optional expert bias."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_expert_bias = args.use_expert_bias

        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, self.num_experts, bias=False)
        if self.use_expert_bias:
            self.expert_bias = mx.zeros((self.num_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x).astype(mx.float32)
        gates = mx.softmax(gates, axis=-1)
        if self.use_expert_bias:
            gates = gates + self.expert_bias

        inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = scores.astype(x.dtype)

        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class Lfm2DecoderLayer(nn.Module):
    """Single decoder layer: Attention or ShortConv, then MLP (dense) or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)

        if layer_idx < args.num_dense_layers:
            self.feed_forward = MLP(args, args.intermediate_size)
        else:
            self.feed_forward = Lfm2MoeSparseMoeBlock(args)

        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        if self.is_attention_layer:
            r = self.self_attn(self.operator_norm(x), mask=mask, cache=cache)
        else:
            r = self.conv(self.operator_norm(x), mask=mask, cache=cache)
        h = x + r
        return h + self.feed_forward(self.ffn_norm(h))


class Lfm2Model(nn.Module):
    """LFM2 MoE transformer backbone (embed + alternating conv/attn + dense/MoE FFN)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self._fa_idx = args.full_attn_idxs[0] if args.full_attn_idxs else 0
        self._conv_idx = next(
            (i for i in range(args.num_hidden_layers) if i not in args.full_attn_idxs),
            0,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        attn_mask = create_attention_mask(h, cache[self._fa_idx])
        conv_mask = create_ssm_mask(h, cache[self._conv_idx])

        for layer, c in zip(self.layers, cache, strict=True):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask=mask, cache=c)

        return self.embedding_norm(h)


class Model(nn.Module):
    """LFM2 MoE top-level; implements ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm2Model(args)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | ArraysCache] | None = None,
        mask: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        return self.model.embed_tokens.as_linear(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return [
            KVCache() if layer.is_attention_layer else ArraysCache(size=1)
            for layer in self.model.layers
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        replacements = {
            "w1.weight": "gate_proj.weight",
            "w2.weight": "down_proj.weight",
            "w3.weight": "up_proj.weight",
        }
        for name, param in weights.items():
            if "conv.weight" in name and param.shape[-1] > param.shape[1]:
                param = param.transpose(0, 2, 1)
            for old, new in replacements.items():
                if old in name:
                    name = name.replace(old, new)
            out[name] = param

        for layer_idx in range(self.args.num_hidden_layers):
            if layer_idx < self.args.num_dense_layers:
                continue
            prefix = f"model.layers.{layer_idx}.feed_forward"
            for proj in ("gate_proj", "down_proj", "up_proj"):
                key = f"{prefix}.experts.0.{proj}.weight"
                if key not in out:
                    continue
                to_join = [
                    out.pop(f"{prefix}.experts.{e}.{proj}.weight")
                    for e in range(self.args.num_experts)
                ]
                out[f"{prefix}.switch_mlp.{proj}.weight"] = mx.stack(to_join)
        return out

    def parameters(self) -> dict[str, Any]:
        return dict(self.trainable_parameters())
