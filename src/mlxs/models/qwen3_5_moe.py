"""Qwen3.5 MoE model: hybrid linear + attention layers with sparse MoE.

Implements ModelProtocol. Config via model_type + text_config (same as mlx_lm).
Uses GatedDeltaNet (linear/SSM) every N layers and Qwen3Next-style attention
elsewhere; MoE blocks use SwitchGLU + optional shared expert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask, create_ssm_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.gated_delta import gated_delta_update
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class TextModelArgs(BaseModelArgs):
    """Qwen3.5 text (language) model config from text_config."""

    model_type: str = "qwen3_5_moe"
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: int | None = None
    full_attention_interval: int = 4
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    mlp_only_layers: list[int] = field(default_factory=list)
    rope_parameters: dict[str, Any] | None = field(
        default_factory=lambda: {
            "type": "default",
            "rope_theta": 100000.0,
            "partial_rotary_factor": 0.25,
        }
    )
    rope_theta: float = 100000.0
    partial_rotary_factor: float = 0.25
    rope_scaling: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters:
            if "rope_type" in self.rope_parameters and "type" not in self.rope_parameters:
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")
            self.partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", 0.25)
            self.rope_theta = float(self.rope_parameters.get("rope_theta", 100000.0))
            self.rope_scaling = self.rope_parameters


def _precise_swiglu_gate(h: mx.array, gate: mx.array, x: mx.array) -> mx.array:
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)


class RMSNormGated(nn.Module):
    """RMSNorm with optional gate (SwiGLU on normalized output)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        gate: mx.array | None = None,
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu_gate(hidden_states, gate, x)
        return x.astype(hidden_states.dtype)


class Attention(nn.Module):
    """Qwen3Next-style attention: QK-norm, RoPE, output gate."""

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim or (args.hidden_size // args.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        rope_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope = initialize_rope(
            rope_dims,
            base=args.rope_theta,
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

        q_proj_out = self.q_proj(x)
        q_proj_out = q_proj_out.reshape(B, L, self.num_attention_heads, -1)
        queries, gate = mx.split(q_proj_out, 2, axis=-1)
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

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
        return self.o_proj(output * mx.sigmoid(gate))


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class GatedDeltaNet(nn.Module):
    """Linear attention / SSM block (Qwen3.5 style): conv + gated delta update."""

    def __init__(self, config: TextModelArgs) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_k_heads")

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones(self.num_v_heads)
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]

        state = cache[1] if cache is not None else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, new_state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask, use_kernel=False
        )
        if cache is not None:
            cache[1] = new_state

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class SparseMoeBlock(nn.Module):
    """Sparse MoE with top-k routing, SwitchGLU experts, and shared expert."""

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_size = args.shared_expert_intermediate_size or args.intermediate_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, self.num_experts)
        self.shared_expert = MLP(dim, shared_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y


class DecoderLayer(nn.Module):
    """One decoder layer: either GatedDeltaNet (linear) or Attention + MLP or MoE."""

    def __init__(self, args: TextModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        is_moe = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.mlp: SparseMoeBlock | MLP = (
            SparseMoeBlock(args) if is_moe else MLP(args.hidden_size, args.intermediate_size)
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ArraysCache | None = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class TextModel(nn.Module):
    """Qwen3.5 text backbone: embed + hybrid layers + norm."""

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache, strict=True):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen3.5 MoE top-level config: model_type + text_config."""

    model_type: str = "qwen3_5_moe"
    text_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        if "text_config" not in params:
            return cls(
                model_type=params.get("model_type", "qwen3_5_moe"),
                text_config=params,
            )
        return cls(**{k: v for k, v in params.items() if k in ("model_type", "text_config")})


class Model(nn.Module):
    """Qwen3.5 MoE: language model wrapper, satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = TextModelArgs.from_dict(args.text_config)
        self.language_model = _TextModelWithHead(text_args)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.language_model(inputs, cache=cache)

    @property
    def num_layers(self) -> int:
        return len(self.language_model.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.language_model.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return [
            ArraysCache(size=2) if layer.is_linear else KVCache()
            for layer in self.language_model.model.layers
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        new_weights: dict[str, Any] = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            new_weights[key] = value

        args = self.language_model.args
        for layer_idx in range(args.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer_idx}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in new_weights:
                gate_up = new_weights.pop(gate_up_key)
                mid = gate_up.shape[-2] // 2
                new_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
                new_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
                down_key = f"{prefix}.experts.down_proj"
                if down_key in new_weights:
                    new_weights[f"{prefix}.switch_mlp.down_proj.weight"] = new_weights.pop(
                        down_key
                    )

        return _text_model_sanitize(new_weights, args)


class _TextModelWithHead(nn.Module):
    """Text model + lm_head (for tie_word_embeddings)."""

    def __init__(self, args: TextModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model = TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)


def _text_model_sanitize(weights: dict[str, Any], args: TextModelArgs) -> dict[str, Any]:
    """Norm shift and conv1d axis fix for language model weights."""
    has_mtp = any("mtp." in k for k in weights)
    has_conv = any("conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items())
    should_shift = has_mtp or has_conv
    weights = {k: v for k, v in weights.items() if "mtp." not in k}

    if args.tie_word_embeddings:
        weights.pop("language_model.lm_head.weight", None)

    norm_keys = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for k, v in list(weights.items()):
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
        if should_shift and any(k.endswith(sfx) for sfx in norm_keys) and v.ndim == 1:
            weights[k] = v + 1.0
    return weights
