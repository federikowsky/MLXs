"""Gemma 2 model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports GQA, RoPE, attention logit soft-capping, final logit
soft-capping, and post-attention / post-feedforward layernorms.

Key differences from Gemma 1:
- ``query_pre_attn_scalar`` replaces head_dim for attention scaling.
- Attention logit soft-capping via tanh before softmax.
- Final logit soft-capping on LM head output.
- Extra pre/post-feedforward layernorms in the transformer block.
- Uses gelu_approx instead of gelu for MLP activation.
- Custom manual SDPA (not mx.fast.scaled_dot_product_attention) due to
  the soft-capping step inserted between score computation and softmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Gemma 2 model configuration."""

    model_type: str = "gemma2"
    hidden_size: int = 3072
    num_hidden_layers: int = 28
    intermediate_size: int = 24576
    num_attention_heads: int = 16
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 256000
    num_key_value_heads: int = 16
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    query_pre_attn_scalar: float = 144.0


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm: applies (1 + weight) scaling."""

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    """Multi-head attention with GQA, RoPE, and logit soft-capping.

    Gemma 2 applies tanh-based soft-capping to attention logits before
    softmax, which requires manual SDPA rather than the fused kernel.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim

        self.scale = 1.0 / (args.query_pre_attn_scalar**0.5)
        self.attn_logit_softcapping = args.attn_logit_softcapping

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
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

        queries = queries * self.scale

        # Manual SDPA with soft-capping (cannot use fused kernel).
        if self.repeats > 1:
            queries = queries.reshape(
                B, self.n_kv_heads, self.repeats, L, self.head_dim
            )
            keys = mx.expand_dims(keys, 2)
            values = mx.expand_dims(values, 2)

        scores = queries @ keys.swapaxes(-1, -2)
        scores = mx.tanh(scores / self.attn_logit_softcapping)
        scores = scores * self.attn_logit_softcapping

        if mask is not None:
            if mask.dtype == mx.bool_:
                scores = mx.where(
                    mask,
                    scores,
                    mx.array(mx.finfo(scores.dtype).min, scores.dtype),
                )
            else:
                scores = scores + mask

        scores = mx.softmax(scores, precise=True, axis=-1)
        output = scores @ values

        if self.repeats > 1:
            output = output.reshape(B, self.n_heads, L, self.head_dim)

        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """Gated MLP with approximate GELU activation (Gemma 2-specific)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with post-attention and post-feedforward norms.

    Gemma 2 adds extra RMSNorm layers after the attention residual and
    after the MLP residual (pre_feedforward_layernorm, post_feedforward_layernorm).
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class Gemma2Model(nn.Module):
    """Gemma 2 transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0], return_array=True)

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """Gemma 2 LM head wrapper -- satisfies ModelProtocol (AC17).

    Gemma 2 ties word embeddings and applies final logit soft-capping
    (tanh-based) on the LM head output.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.final_logit_softcapping = args.final_logit_softcapping
        self.model = Gemma2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
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
        return {
            k: v
            for k, v in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.layers
