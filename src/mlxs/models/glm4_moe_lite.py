"""GLM-4 MoE Lite — port from mlx_lm (glm4_moe_lite.py), ModelProtocol-compliant.

MLA-style attention (Q/KV LoRA, MultiLinear, RoPE on subset of heads), MoE with
group_expert_select, optional shared experts. Imports only from mlxs.cache,
mlxs.layers, mlxs.models.base.
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
from mlxs.layers.mla import MultiLinear
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs

# ----- Cache for MLA-style attention (kv_latent + k_pe) -----


class Glm4MoeLiteKVCache(KVCache):
    """KV cache for GLM-4 MoE Lite: stores kv_latent (B,1,S,kv_rank) and k_pe (B,1,S,rope_dim).

    update_and_fetch(keys, values) receives (kv_latent, k_pe) and returns
    (full_kv_latent, full_k_pe). Satisfies CacheProtocol.
    """

    def __init__(self, kv_rank: int, rope_dim: int) -> None:
        super().__init__()
        self._kv_rank = kv_rank
        self._rope_dim = rope_dim

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # keys = kv_latent (B, 1, T, kv_rank), values = k_pe (B, 1, T, rope_dim)
        prev = self._offset
        n_new = keys.shape[2]

        if self._keys is None or (prev + n_new) > self._keys.shape[2]:
            self._grow_mla(keys, values, prev, n_new)

        self._offset = prev + n_new
        self._keys[..., prev : self._offset, :] = keys
        self._values[..., prev : self._offset, :] = values
        return (
            self._keys[..., : self._offset, :],
            self._values[..., : self._offset, :],
        )

    def _grow_mla(
        self,
        keys: mx.array,
        values: mx.array,
        prev: int,
        n_new: int,
    ) -> None:
        B = keys.shape[0]
        kv_rank = keys.shape[3]
        rope_dim = values.shape[3]
        n_steps = max(1, (self.step + n_new - 1) // self.step)
        cap = n_steps * self.step
        k_shape = (B, 1, cap, kv_rank)
        v_shape = (B, 1, cap, rope_dim)
        new_k = mx.zeros(k_shape, keys.dtype)
        new_v = mx.zeros(v_shape, values.dtype)
        if self._keys is not None:
            if prev % self.step != 0:
                self._keys = self._keys[..., :prev, :]
                self._values = self._values[..., :prev, :]
            self._keys = mx.concatenate([self._keys, new_k], axis=2)
            self._values = mx.concatenate([self._values, new_v], axis=2)
        else:
            self._keys, self._values = new_k, new_v


# ----- Model args -----


@dataclass
class ModelArgs(BaseModelArgs):
    """GLM-4 MoE Lite config (config.json)."""

    model_type: str = "glm4_moe_lite"
    vocab_size: int = 154880
    hidden_size: int = 2048
    intermediate_size: int = 10240
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 47
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    n_shared_experts: int | None = 1
    n_routed_experts: int | None = 64
    routed_scaling_factor: float = 1.8
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 768
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 192
    v_head_dim: int = 256
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    partial_rotary_factor: float = 1.0
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 1


# ----- MoE router (no mx.compile for testability) -----


def _group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[mx.array, mx.array]:
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


# ----- Attention -----


class Glm4MoeLiteAttention(nn.Module):
    """MLA-style attention: Q/KV LoRA, MultiLinear, RoPE on qk_rope_head_dim."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.scale = self.q_head_dim**-0.5

        rope_params = args.rope_scaling
        if rope_params is not None and rope_params.get("mscale_all_dim", 0):
            mscale_all_dim = rope_params["mscale_all_dim"]
            scaling_factor = rope_params.get("factor", 1.0)
            if scaling_factor > 1:
                s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.scale = self.scale * s * s

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=args.attention_bias,
            )
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=args.attention_bias)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_head_dim, self.num_heads)
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=args.attention_bias,
        )

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=rope_params,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Glm4MoeLiteKVCache | None = None,
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
        kv_latent = self.kv_a_layernorm(compressed_kv)

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


# ----- MLP -----


class Glm4MoeLiteMLP(nn.Module):
    """SwiGLU MLP (dense or shared expert)."""

    def __init__(
        self,
        args: ModelArgs,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        h = args.hidden_size if hidden_size is None else hidden_size
        i = args.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


# ----- MoE gate -----


class MoEGate(nn.Module):
    """Router: weight (n_routed_experts, hidden_size), e_score_correction_bias."""

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
        return _group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


# ----- MoE block -----


class Glm4MoeLiteMoE(nn.Module):
    """MoE: MoEGate + SwitchGLU experts + optional shared expert MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
        )
        self.gate = MoEGate(args)
        self.shared_experts: Glm4MoeLiteMLP | None = None
        if args.n_shared_experts is not None and args.n_shared_experts > 0:
            self.shared_experts = Glm4MoeLiteMLP(
                args,
                intermediate_size=args.moe_intermediate_size * args.n_shared_experts,
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


# ----- Decoder layer -----


class Glm4MoeLiteDecoderLayer(nn.Module):
    """Pre-norm attention + residual; pre-norm MLP or MoE + residual."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Glm4MoeLiteAttention(args)
        use_moe = (
            args.n_routed_experts is not None
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        )
        self.mlp: Glm4MoeLiteMoE | Glm4MoeLiteMLP = (
            Glm4MoeLiteMoE(args) if use_moe else Glm4MoeLiteMLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Glm4MoeLiteKVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# ----- Transformer -----


class Glm4MoeLiteModel(nn.Module):
    """Transformer: embed, decoder layers, final norm. No pipeline/distributed."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Glm4MoeLiteDecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: list[Glm4MoeLiteKVCache] | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        h = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        if mask is None:
            mask = create_attention_mask(h, cache[0], return_array=True)
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)
        return self.norm(h)


# ----- Model (ModelProtocol) -----


class Model(nn.Module):
    """GLM-4 MoE Lite LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm4MoeLiteModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[Glm4MoeLiteKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache, mask=mask)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[Glm4MoeLiteKVCache]:
        args = self.args
        return [
            Glm4MoeLiteKVCache(args.kv_lora_rank, args.qk_rope_head_dim)
            for _ in range(self.num_layers)
        ]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        def is_mpt_layer(key: str) -> bool:
            parts = key.split(".")
            return (
                len(parts) >= 3
                and parts[1] == "layers"
                and int(parts[2]) >= self.args.num_hidden_layers
            )

        new_weights = {k: v for k, v in weights.items() if not is_mpt_layer(k)}
        weights = new_weights

        n_experts = self.args.n_routed_experts or 0
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for _hf, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(n_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
            prefix_attn = f"model.layers.{layer_idx}.self_attn"
            if f"{prefix_attn}.kv_b_proj.weight" in weights:
                v = weights.pop(f"{prefix_attn}.kv_b_proj.weight")
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                num_heads = self.args.num_attention_heads
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
                weights[f"{prefix_attn}.embed_q.weight"] = wk
                weights[f"{prefix_attn}.unembed_out.weight"] = wv

        return weights
