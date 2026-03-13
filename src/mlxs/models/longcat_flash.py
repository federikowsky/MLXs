"""Longcat Flash model — MLA + dual-attn block + MoE (SwitchGLU), ModelProtocol-compliant.

Ported from mlx_lm longcat_flash. Uses dual attention blocks per layer, RoPE on q/k_pe,
MultiLinear (embed_q / unembed_out) for compressed KV, and MoE with identity zero expert.
Imports: mlxs.cache, mlxs.layers, mlxs.models.base. Norms: nn.RMSNorm only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.cache_list import CacheList
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.mla import MultiLinear
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Longcat Flash config; from_dict aligned to config.json."""

    model_type: str = "longcat_flash"
    attention_method: str = "flash"
    zero_expert_type: str = "identity"
    hidden_size: int = 2048
    ffn_hidden_size: int = 5632
    moe_topk: int = 1
    expert_ffn_hidden_size: int = 1408
    n_routed_experts: int = 8
    zero_expert_num: int = 1
    num_layers: int = 32
    vocab_size: int = 128256
    max_position_embeddings: int = 32768
    num_attention_heads: int = 16
    kv_lora_rank: int = 256
    q_lora_rank: int | None = 768
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 128
    routed_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    mla_scale_q_lora: bool = False
    mla_scale_kv_lora: bool = False
    attention_bias: bool = False
    norm_topk_prob: bool = False
    router_bias: bool = False
    rope_scaling: dict[str, Any] | None = None


class LongcatFlashMLA(nn.Module):
    """MLA attention: q (optional LoRA), kv_latent + k_pe, RoPE on pe, embed_q/unembed_out."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.v_head_dim = args.v_head_dim

        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.scale = self.qk_head_dim**-0.5
        self.mla_scale_q_lora: float | None = None
        self.mla_scale_kv_lora: float | None = None

        if args.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size,
                self.num_attention_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=args.attention_bias)
            self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank)
            self.q_b_proj = nn.Linear(
                args.q_lora_rank,
                self.num_attention_heads * self.qk_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_attention_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_attention_heads
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * args.v_head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        if args.mla_scale_q_lora and args.q_lora_rank is not None:
            self.mla_scale_q_lora = (args.hidden_size / args.q_lora_rank) ** 0.5
        if args.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (args.hidden_size / self.kv_lora_rank) ** 0.5

        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = args.rope_scaling.get("factor", 1.0)
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_attention_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        if self.mla_scale_q_lora is not None:
            q = q * self.mla_scale_q_lora

        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        if self.mla_scale_kv_lora is not None:
            kv_latent = kv_latent * self.mla_scale_kv_lora

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=None, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class LongcatFlashMLP(nn.Module):
    """Dense SwiGLU MLP (gate/up/down)."""

    def __init__(self, args: ModelArgs, is_expert: bool = False) -> None:
        super().__init__()
        hidden_size = args.expert_ffn_hidden_size if is_expert else args.ffn_hidden_size
        self.gate_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LongcatFlashTopkRouter(nn.Module):
    """Top-k router with optional norm_topk_prob and routed_scaling_factor."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.top_k = args.moe_topk
        self.n_routed_experts = args.n_routed_experts + args.zero_expert_num
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.router_bias = args.router_bias
        self.n_regular = args.n_routed_experts

        self.classifier = nn.Linear(args.hidden_size, self.n_routed_experts, bias=self.router_bias)
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        dtype = hidden_states.dtype
        router_logits = self.classifier(hidden_states)
        scores = mx.softmax(router_logits, axis=-1)
        corrected_scores = scores + self.e_score_correction_bias
        topk_indices = mx.argpartition(corrected_scores, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)
        if self.norm_topk_prob:
            denominator = mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights.astype(dtype)


class LongcatFlashMoE(nn.Module):
    """MoE with SwitchGLU experts and identity zero expert."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_routed_experts = args.n_routed_experts
        self.zero_expert_num = args.zero_expert_num
        self.zero_expert_type = args.zero_expert_type

        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.expert_ffn_hidden_size,
            args.n_routed_experts,
        )
        self.router = LongcatFlashTopkRouter(args)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        topk_indices, topk_weights = self.router(hidden_states)
        mask = topk_indices >= self.n_routed_experts
        topk_indices = mx.where(mask, 0, topk_indices)
        regular_weights = mx.where(mask, 0.0, topk_weights)

        regular_outputs = self.switch_mlp(hidden_states, topk_indices)
        weighted_outputs = regular_outputs * regular_weights[..., None]
        final_output = mx.sum(weighted_outputs, axis=-2)

        if self.zero_expert_type == "identity":
            identity_weights_sum = mx.sum(
                mx.where(mask, topk_weights, 0.0), axis=-1, keepdims=True
            )
            final_output = final_output + hidden_states * identity_weights_sum

        return final_output


class LongcatFlashDecoderLayer(nn.Module):
    """Dual attention + MoE after first attn + two MLPs; shortcut adds MoE to second."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.mlp = LongcatFlashMoE(args)
        self.self_attn = [LongcatFlashMLA(args) for _ in range(2)]
        self.mlps = [LongcatFlashMLP(args, is_expert=False) for _ in range(2)]
        self.input_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)
        ]
        self.post_attention_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)
        ]

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[KVCache | None, KVCache | None] | None = None,
    ) -> mx.array:
        hidden_states = x
        shortcut_mlp_output: mx.array | None = None

        if cache is None:
            cache = (None, None)

        for i in range(2):
            residual = hidden_states
            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states = self.self_attn[i](hidden_states, mask=mask, cache=cache[i])
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm[i](hidden_states)
            if i == 0:
                shortcut_mlp_output = self.mlp(hidden_states)
            hidden_states = self.mlps[i](hidden_states)
            hidden_states = residual + hidden_states
            if i == 1 and shortcut_mlp_output is not None:
                hidden_states = hidden_states + shortcut_mlp_output

        return hidden_states


class LongcatFlashModel(nn.Module):
    """Embedding + stacked decoder layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_layers = args.num_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [LongcatFlashDecoderLayer(args) for _ in range(args.num_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: list[CacheList] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(x)
        if cache is None:
            cache = [None] * self.num_layers  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0], return_array=True)

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask=mask, cache=(c[0], c[1]) if c is not None else None)

        return self.norm(h)


class Model(nn.Module):
    """Longcat Flash LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LongcatFlashModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[CacheList] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        del mask
        return self.lm_head(self.model(input_ids, cache=cache))

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[CacheList]:
        return [CacheList(KVCache(), KVCache(), kv_index=0) for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Remap experts and kv_b_proj to embed_q/unembed_out (mlx_lm-compatible)."""
        for layer_idx in range(self.args.num_layers):
            prefix = f"model.layers.{layer_idx}"
            for _hf, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    key = f"{prefix}.mlp.experts.0.{m}.{k}"
                    if key in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        for layer_idx in range(self.args.num_layers):
            for i in range(2):
                prefix = f"model.layers.{layer_idx}.self_attn.{i}"
                kv_b_key = f"{prefix}.kv_b_proj.weight"
                if kv_b_key not in weights:
                    continue
                num_heads = self.args.num_attention_heads
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                quantized = f"{prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{prefix}.embed_q.scales"] = wk_s
                    weights[f"{prefix}.embed_q.biases"] = wk_b
                    weights[f"{prefix}.unembed_out.scales"] = wv_s
                    weights[f"{prefix}.unembed_out.biases"] = wv_b

                weights[f"{prefix}.embed_q.weight"] = wk
                weights[f"{prefix}.unembed_out.weight"] = wv

        return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}
