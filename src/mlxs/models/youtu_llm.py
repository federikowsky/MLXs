"""YoutuLLM model — port from mlx_lm, ModelProtocol-compliant.

Text backbone only (VL/multimodal: image path out of scope). Architecture:
q LoRA (q_a_proj → RMSNorm → q_b_proj) and kv LoRA with MQA-style k_pe;
RoPE on qk_rope_head_dim only; standard RMSNorm and SwiGLU MLP.
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
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """YoutuLLM config; from_dict aligned to config.json."""

    model_type: str = "youtu_llm"
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1600000.0
    rope_traditional: bool = True
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True


class Attention(nn.Module):
    """YoutuLLM attention: q LoRA (optional), kv LoRA with MQA-style k_pe, RoPE on rope dims."""

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

        if args.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size,
                args.q_lora_rank,
                bias=args.attention_bias,
            )
            self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                args.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
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

        self.rope = initialize_rope(
            self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
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
            q_pe = self.rope(q_pe, offset=cache.offset)
            k_pe = self.rope(k_pe, offset=cache.offset)
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)
        k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
        keys = mx.concatenate([k_nope, k_pe], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        output = scaled_dot_product_attention(queries, keys, values, cache, self.scale, mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP with optional bias."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
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


class YoutuLLMModel(nn.Module):
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
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0] if cache else None)
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    """YoutuLLM Model — ModelProtocol-compliant, text-only path."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = YoutuLLMModel(args)
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
        if self.args.tie_word_embeddings:
            out = out @ self.model.embed_tokens.weight.T
        else:
            out = self.lm_head(out)
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
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights
