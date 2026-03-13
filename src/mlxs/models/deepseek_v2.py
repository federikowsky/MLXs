"""DeepSeek V2 model architecture with MLA and sparse MoE.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports Multi-head Latent Attention (MLA) with LoRA-compressed KV,
YarnRoPE, SwiGLU, and sparse MoE with shared experts and SwitchGLU.
"""

from __future__ import annotations

import math
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


# ---------------------------------------------------------------------------
# Yarn helpers (DeepSeek V2 uses its own YarnRoPE variant inline)
# ---------------------------------------------------------------------------

def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs(BaseModelArgs):
    """DeepSeek V2 model configuration."""

    model_type: str = "deepseek_v2"
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: int | None = None
    n_routed_experts: int | None = None
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "gready"
    n_group: int | None = None
    topk_group: int | None = None
    num_experts_per_tok: int | None = None
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False


# ---------------------------------------------------------------------------
# DeepSeek V2 YarnRoPE (custom variant with mscale normalization)
# ---------------------------------------------------------------------------

class DeepseekV2YarnRotaryEmbedding(nn.Module):
    """Yarn RoPE variant used by DeepSeek V2 attention."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ) -> None:
        super().__init__()
        self.mscale = _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )

        def _find_correction_dim(num_rotations: float) -> float:
            return (
                dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        def _find_correction_range() -> tuple[int, int]:
            low = math.floor(_find_correction_dim(beta_fast))
            high = math.ceil(_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dim - 1)

        def _linear_ramp_mask(min_val: int, max_val: int, d: int) -> mx.array:
            if min_val == max_val:
                max_val += 0.001
            linear_func = (mx.arange(d, dtype=mx.float32) - min_val) / (max_val - min_val)
            return mx.clip(linear_func, 0, 1)

        freq_extra = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        freq_inter = scaling_factor * base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        low, high = _find_correction_range()
        freq_mask = 1.0 - _linear_ramp_mask(low, high, dim // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        if self.mscale != 1.0:
            x = self.mscale * x
        return mx.fast.rope(
            x,
            x.shape[-1],
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


# ---------------------------------------------------------------------------
# Multi-head Latent Attention (MLA)
# ---------------------------------------------------------------------------

class DeepseekV2Attention(nn.Module):
    """MLA with LoRA-compressed KV and YarnRoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.rope_theta = args.rope_theta
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=args.attention_bias,
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=args.attention_bias,
        )

        # Build RoPE with mscale adjustment
        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = args.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scale = self.scale * mscale * mscale

            rope_kwargs = {
                key: args.rope_scaling[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in args.rope_scaling
            }
            self.rope = DeepseekV2YarnRotaryEmbedding(
                dim=self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
                **rope_kwargs,
            )
        else:
            self.rope = initialize_rope(
                self.qk_rope_head_dim,
                base=self.rope_theta,
                traditional=True,
                max_position_embeddings=self.max_position_embeddings,
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        if cache is not None:
            q_pe = self.rope(q_pe, cache.offset)
            k_pe = self.rope(k_pe, cache.offset)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys, values = cache.update_and_fetch(
                mx.concatenate([k_nope, k_pe], axis=-1), values,
            )
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys = mx.concatenate([k_nope, k_pe], axis=-1)

        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


# ---------------------------------------------------------------------------
# MLP / MoE
# ---------------------------------------------------------------------------

class DeepseekV2MLP(nn.Module):
    """SwiGLU MLP with configurable hidden/intermediate sizes."""

    def __init__(
        self,
        args: ModelArgs,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size if hidden_size is not None else args.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else args.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    """Top-k expert gating with optional group-limited greedy routing."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.topk_method = args.topk_method
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = mx.zeros((self.n_routed_experts, args.hidden_size))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        gates = x @ self.weight.T
        scores = mx.softmax(gates, axis=-1, precise=True)

        if self.topk_method == "group_limited_greedy":
            bsz, seq_len = x.shape[:2]
            scores = scores.reshape(bsz, seq_len, self.n_group, -1)
            group_scores = scores.max(axis=-1, keepdims=True)
            k = self.n_group - self.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            scores = mx.put_along_axis(
                scores, group_idx, mx.array(0.0, scores.dtype), axis=-2,
            )
            scores = scores.reshape(bsz, seq_len, -1)

        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(scores, inds, axis=-1)
        scores = scores * self.routed_scaling_factor
        return inds, scores


class DeepseekV2MoE(nn.Module):
    """Sparse MoE block with optional shared experts."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self._n_shared_experts = args.n_shared_experts
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
        )
        self.gate = MoEGate(args)
        if args.n_shared_experts is not None:
            intermediate_size = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                args, intermediate_size=intermediate_size,
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self._n_shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


# ---------------------------------------------------------------------------
# Decoder layer / backbone / Model
# ---------------------------------------------------------------------------

class DeepseekV2DecoderLayer(nn.Module):
    """Pre-norm transformer block with dense MLP or MoE."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = DeepseekV2Attention(args)
        self.mlp: DeepseekV2MoE | DeepseekV2MLP = (
            DeepseekV2MoE(args)
            if (
                args.n_routed_experts is not None
                and layer_idx >= args.first_k_dense_replace
                and layer_idx % args.moe_layer_freq == 0
            )
            else DeepseekV2MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DeepseekV2Backbone(nn.Module):
    """DeepSeek V2 transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DeepseekV2DecoderLayer(args, idx)
            for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """DeepSeek V2 LM head wrapper -- satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV2Backbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
        return weights

    @property
    def layers(self) -> list[DeepseekV2DecoderLayer]:
        return self.model.layers
