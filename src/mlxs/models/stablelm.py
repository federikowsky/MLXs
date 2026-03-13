"""StableLM model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports partial rotary embeddings, optional QK layernorm, and parallel residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache
from mlxs.models.activations import swiglu
from mlxs.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)


@dataclass
class ModelArgs(BaseModelArgs):
    """StableLM model configuration."""

    model_type: str = "stablelm"
    vocab_size: int = 50304
    hidden_size: int = 6144
    num_attention_heads: int = 48
    num_hidden_layers: int = 44
    num_key_value_heads: int = 8
    intermediate_size: int = 16384
    rope_theta: float = 10000.0
    use_qkv_bias: bool = False
    partial_rotary_factor: float = 0.25
    layer_norm_eps: float = 1e-5
    use_parallel_residual: bool = False
    qk_layernorm: bool = False


class LayerNormPerHead(nn.Module):
    """Per-head layer normalization for query/key projections."""

    def __init__(self, head_dim: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.norms = [
            nn.LayerNorm(head_dim, eps=eps, bias=False) for _ in range(num_heads)
        ]
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        w = mx.stack([n.weight for n in self.norms])
        return w * mx.fast.layer_norm(x, None, None, self.eps)


class Attention(nn.Module):
    """Multi-head attention with partial RoPE and optional QK layernorm."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.use_qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rope = nn.RoPE(
            int(self.partial_rotary_factor * self.head_dim),
            traditional=False,
            base=self.rope_theta,
        )

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = LayerNormPerHead(
                self.head_dim, self.num_heads, eps=config.layer_norm_eps
            )
            self.k_layernorm = LayerNormPerHead(
                self.head_dim, self.num_key_value_heads, eps=config.layer_norm_eps
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        B, L, _ = queries.shape

        queries = queries.reshape(B, L, self.num_heads, -1)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1)
        if self.qk_layernorm:
            queries = self.q_layernorm(queries)
            keys = self.k_layernorm(keys)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        scale = math.sqrt(1 / queries.shape[-1])
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        ).astype(values.dtype)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    """StableLM decoder layer with optional parallel residual."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(config=config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.use_parallel_residual = config.use_parallel_residual
        if not self.use_parallel_residual:
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        r = self.self_attn(h, mask, cache)

        if self.use_parallel_residual:
            out = x + r + self.mlp(h)
        else:
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))
            out = h + r
        return out


class StableLM(nn.Module):
    """StableLM transformer backbone."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        x = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        mask = create_attention_mask(x, cache[0])

        for layer, c in zip(self.layers, cache):
            x = layer(x, mask, cache=c)

        return self.norm(x)


class Model(nn.Module):
    """StableLM LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = StableLM(args)
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
        return {
            k: v
            for k, v in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers
