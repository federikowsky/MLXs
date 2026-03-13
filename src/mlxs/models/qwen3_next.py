"""Qwen3-Next hybrid model (linear/SSM + full attention + MoE) — ModelProtocol.

Port from mlx_lm. Alternating GatedDeltaNet (linear) and full attention every
full_attention_interval; sparse MoE or dense MLP per decoder_sparse_step.
Uses ArraysCache(size=2) for linear layers, KVCache for attention.
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
from mlxs.layers.gated_delta import gated_delta_update
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs

# ----- Model args -----


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen3-Next text model configuration (flat, from config.json)."""

    model_type: str = "qwen3_next"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    mlp_only_layers: tuple[int, ...] = ()
    moe_intermediate_size: int = 14336
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    rope_theta: float = 100000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 131072
    head_dim: int = 128
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    rope_scaling: dict[str, Any] | None = None
    full_attention_interval: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.mlp_only_layers, list):
            object.__setattr__(self, "mlp_only_layers", tuple(self.mlp_only_layers))


# ----- RMSNormGated (model-specific) -----


class RMSNormGated(nn.Module):
    """RMSNorm then optional gate: out = sigmoid(gate) * rms_norm(x) or x."""

    def __init__(self, dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array, gate: mx.array | None = None) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return (nn.silu(gate.astype(mx.float32)) * x.astype(mx.float32)).astype(
                hidden_states.dtype
            )
        return x.astype(hidden_states.dtype)


# ----- Attention -----


class Attention(nn.Module):
    """Full attention with QK-norm, partial RoPE, and output gate (sigmoid)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        rope_dims = int(self.head_dim * args.partial_rotary_factor)
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
        queries, gate = mx.split(
            q_proj_out.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
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


# ----- MLP -----


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


# ----- GatedDeltaNet (qwen3_next: single qkvz + ba projection) -----


def _fix_query_key_value_ordering(
    mixed_qkvz: mx.array,
    mixed_ba: mx.array,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Split combined qkvz and ba projections into q, k, v, z, b, a."""
    mixed_qkvz = mixed_qkvz.reshape(*mixed_qkvz.shape[:-1], num_k_heads, -1)
    mixed_ba = mixed_ba.reshape(*mixed_ba.shape[:-1], num_k_heads, -1)
    nv_per_k = num_v_heads // num_k_heads
    q, k, v, z = mx.split(
        mixed_qkvz,
        [head_k_dim, 2 * head_k_dim, 2 * head_k_dim + nv_per_k * head_v_dim],
        axis=-1,
    )
    b, a = mx.split(mixed_ba, [nv_per_k], axis=-1)
    return (
        q,
        k,
        v.reshape(*v.shape[:2], -1, head_v_dim),
        z.reshape(*z.shape[:2], -1, head_v_dim),
        b.reshape(*b.shape[:2], num_v_heads),
        a.reshape(*a.shape[:2], num_v_heads),
    )


class GatedDeltaNet(nn.Module):
    """Linear attention block: single qkvz/ba projection, conv, gated delta recurrence."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"linear_num_value_heads ({self.num_v_heads}) must be divisible "
                f"by linear_num_key_heads ({self.num_k_heads})"
            )
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        self.in_proj_qkvz = nn.Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim * 2,
            bias=False,
        )
        self.in_proj_ba = nn.Linear(self.hidden_size, self.num_v_heads * 2, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,))
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)
        self.norm = RMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        mixed_qkvz = self.in_proj_qkvz(inputs)
        mixed_ba = self.in_proj_ba(inputs)
        q, k, v, z, b, a = _fix_query_key_value_ordering(
            mixed_qkvz,
            mixed_ba,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        mixed_qkv = mx.concatenate(
            [
                q.reshape(B, S, -1),
                k.reshape(B, S, -1),
                v.reshape(B, S, -1),
            ],
            axis=-1,
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        n_keep = self.conv_kernel_size - 1
        if cache is not None:
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = conv_input[:, -n_keep:, :]
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1),
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
            cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


# ----- Sparse MoE -----


class SparseMoeBlock(nn.Module):
    """Sparse MoE: gate, top-k, SwitchGLU experts, shared expert."""

    def __init__(self, args: ModelArgs) -> None:
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


# ----- Decoder layer -----


class DecoderLayer(nn.Module):
    """One block: GatedDeltaNet or Attention, then MLP or SparseMoeBlock."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
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


# ----- Backbone and Model -----


class Qwen3NextModel(nn.Module):
    """Qwen3-Next backbone: embed -> alternating linear/attn layers -> norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._ssm_idx = 0
        self._fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ArraysCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        fa_mask = create_attention_mask(h, cache[self._fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self._ssm_idx])
        for layer, c in zip(self.layers, cache, strict=True):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """Qwen3-Next LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | ArraysCache] | None = None,
        mask: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | ArraysCache]:
        return [
            ArraysCache(size=2) if layer.is_linear else KVCache() for layer in self.model.layers
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
        weights = {k: v for k, v in weights.items() if "mtp." not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            for name in ("up_proj", "down_proj", "gate_proj"):
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{name}.weight")
                    for e in range(self.args.num_experts)
                ]
                weights[f"{prefix}.switch_mlp.{name}.weight"] = mx.stack(to_join)
        norm_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if any(k.endswith(sfx) for sfx in norm_suffixes) and v.ndim == 1:
                weights[k] = v + 1.0
        return weights
