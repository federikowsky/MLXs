"""Ernie 4.5 MoE model — port from mlx_lm (ernie4_5_moe.py), ModelProtocol-compliant.

Dense + MoE decoder with RMSNorm, RoPE (traditional), SwitchGLU experts,
optional shared experts, configurable MoE layer range and gate activation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Ernie 4.5 MoE config (config.json)."""

    model_type: str = "ernie4_5_moe"
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 24
    num_key_value_heads: int = 24
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    use_bias: bool = False
    tie_word_embeddings: bool = False
    moe_num_experts: int = 8
    moe_layer_start_index: int | list[int] = 0
    moe_layer_end_index: int | list[int] | None = None
    moe_intermediate_size: int = 0
    moe_capacity: list[int] = field(default_factory=list)
    moe_k: int = 1
    moe_layer_interval: int = 1
    moe_use_aux_free: bool = False
    moe_num_shared_experts: int = 0
    moe_gate_act: str = "softmax"
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class Attention(nn.Module):
    """Multi-head attention with RoPE (traditional), no QK-norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or dim // self.n_heads
        self.scale = self.head_dim**-0.5
        bias = args.use_bias

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=True,
            scaling_config=None,
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
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class ErnieMLP(nn.Module):
    """Dense SwiGLU MLP (non-MoE layers and shared experts)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class ErnieMoeMLP(nn.Module):
    """MoE block: router (gate + top-k) + SwitchGLU experts + optional shared expert."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.k = args.moe_k
        self.moe_intermediate_size = (
            args.moe_intermediate_size if args.moe_intermediate_size else args.intermediate_size
        )

        self.gate = nn.Linear(args.hidden_size, args.moe_num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            self.moe_intermediate_size,
            args.moe_num_experts,
            bias=args.use_bias,
        )

        if getattr(args, "moe_num_shared_experts", 0) > 0:
            shared_intermediate_size = self.moe_intermediate_size * args.moe_num_shared_experts
            self.shared_experts = ErnieMLP(
                args.hidden_size, shared_intermediate_size, args.use_bias
            )
        else:
            self.shared_experts = None

        if args.moe_gate_act == "softmax":
            self.gate_act = nn.Softmax()
        elif args.moe_gate_act == "sigmoid":
            self.gate_act = nn.Sigmoid()
        else:
            raise ValueError(f"moe_gate_act '{args.moe_gate_act}' is not supported")

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = self.gate_act(gates.astype(mx.float32))

        k = self.k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


def _moe_layer_start(args: ModelArgs) -> int:
    v = args.moe_layer_start_index
    return min(v) if isinstance(v, (list, tuple)) else v


def _moe_layer_end(args: ModelArgs) -> int:
    v = args.moe_layer_end_index
    if v is None:
        return args.num_hidden_layers - 1
    return max(v) if isinstance(v, (list, tuple)) else v


class DecoderLayer(nn.Module):
    """One decoder layer: attention + MLP or MoE (by layer index)."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        start = _moe_layer_start(args)
        end = _moe_layer_end(args)
        is_moe = (layer_idx + 1) % args.moe_layer_interval == 0 and start <= layer_idx <= end
        if is_moe:
            self.mlp = ErnieMoeMLP(args)
        else:
            self.mlp = ErnieMLP(args.hidden_size, args.intermediate_size, args.use_bias)
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


class Ernie45Model(nn.Module):
    """Backbone: embed + decoder layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | None] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0] if cache else None)
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    """Ernie 4.5 MoE: language model wrapper, satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Ernie45Model(args)
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
            out = self.model.embed_tokens.as_linear(out)
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
        remove_patterns = [
            "mtp_block.",
            "mtp_linear_proj.",
            "mtp_hidden_norm.",
            "mtp_emb_norm.",
            "e_score_correction_bias",
        ]
        weights = {k: v for k, v in weights.items() if not any(p in k for p in remove_patterns)}
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                key = f"{prefix}.mlp.experts.0.{m}.weight"
                if key in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.weight")
                        for e in range(self.args.moe_num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.weight"] = mx.stack(to_join)
        return weights
