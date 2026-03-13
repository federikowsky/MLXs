"""MiniCPM3 model — port from mlx_lm, ModelProtocol-compliant.

Architecture: q LoRA (q_a_proj → layernorm → q_b_proj) and kv LoRA (kv_a_proj_with_mqa
→ split to compressed_kv + k_pe, kv_a_layernorm → kv_b_proj) for attention;
SuScaledRoPE (longrope) on qk_rope_head_dim only; separate qk_nope / qk_rope / v head dims.
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
from mlxs.layers.rope import SuScaledRoPE
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """MiniCPM3 config; from_dict aligned to config.json (HF MiniCPM3Config)."""

    model_type: str = "minicpm3"
    hidden_size: int = 4096
    dim_model_base: int = 1
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    num_key_value_heads: int | None = None
    q_lora_rank: int = 768
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    kv_lora_rank: int = 256
    v_head_dim: int | None = None
    scale_depth: float = 1.0
    scale_emb: float = 1.0
    max_position_embeddings: int = 2048
    attention_bias: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = True


class Attention(nn.Module):
    """MiniCPM3 attention: q LoRA (q_a → layernorm → q_b), kv LoRA with MQA-style k_pe.

    Query: hidden → q_a_proj (q_lora_rank) → RMSNorm → q_b_proj → [B,L,num_heads,q_head_dim].
    Key/Value: hidden → kv_a_proj_with_mqa (kv_lora_rank + qk_rope_head_dim); split to
    compressed_kv and k_pe; layernorm(compressed_kv) → kv_b_proj → k_nope + values.
    RoPE (SuScaledRoPE) applied only to q_pe and k_pe (qk_rope_head_dim). Matches mlx_lm
    class Attention and formulas.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.num_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = (
            args.v_head_dim
            if args.v_head_dim is not None
            else args.hidden_size // args.num_attention_heads
        )
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=args.attention_bias)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        kv_b_out = self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, kv_b_out, bias=False)

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=args.attention_bias,
        )

        scaling = args.rope_scaling or {}
        self.rope = SuScaledRoPE(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            original_max_position_embeddings=scaling.get("original_max_position_embeddings", 4096),
            short_factor=scaling.get("short_factor", 1.0),
            long_factor=scaling.get("long_factor", 1.0),
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        if cache is not None:
            q_pe = self.rope(q_pe, offset=cache.offset)
            k_pe = self.rope(k_pe, offset=cache.offset)
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)

        k_pe_broadcasted = mx.broadcast_to(k_pe, (B, self.num_heads, L, self.qk_rope_head_dim))
        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        keys = mx.concatenate([k_nope, k_pe_broadcasted], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, self.softmax_scale, mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.num_hidden_layers
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.scale_depth = args.scale_depth

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r * (self.scale_depth / (self.num_hidden_layers**0.5))
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r * (self.scale_depth / (self.num_hidden_layers**0.5))


class MiniCPM3Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) * self.args.scale_emb
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0] if cache else None)
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniCPM3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        if not self.args.tie_word_embeddings:
            out = self.lm_head(out / (self.args.hidden_size / self.args.dim_model_base))
        else:
            out = out @ self.model.embed_tokens.weight.T
        return out

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(len(self.model.layers))]

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        return weights
