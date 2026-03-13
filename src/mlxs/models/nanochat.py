"""NanoChat model architecture.

Implements ModelProtocol. Pre-norm transformer with RoPE, QK-norm (RMSNorm on Q/K),
ReLU² MLP, and logits softcap. Uses functional RMSNorm (no learnable scale).
Compatible with mlx_lm-converted weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import relu_squared
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.models.base import BaseModelArgs

_RMS_EPS = 1e-5
_SOFTCAP = 15.0


def _rms_norm(x: mx.array) -> mx.array:
    """Functional RMSNorm with no learnable parameters (nanochat uses this throughout)."""
    return mx.fast.rms_norm(x, None, _RMS_EPS)


def _apply_rope(
    x: mx.array,
    offset: int,
    freqs: mx.array,
    head_dim: int,
) -> mx.array:
    """Apply RoPE with precomputed negated frequencies (blocked layout, traditional=False)."""
    return mx.fast.rope(
        x,
        dims=head_dim,
        traditional=False,
        base=None,
        freqs=freqs,
        scale=1.0,
        offset=offset,
    )


@partial(mx.compile, shapeless=True)
def _softcap(logits: mx.array, cap: float = _SOFTCAP) -> mx.array:
    """Softcap logits for stability (nanochat default cap=15)."""
    return cap * mx.tanh(logits / cap)


@dataclass
class ModelArgs(BaseModelArgs):
    """NanoChat model configuration."""

    model_type: str = "nanochat"
    hidden_size: int = 1280
    num_hidden_layers: int = 20
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    vocab_size: int = 65536
    max_position_embeddings: int = 2048
    intermediate_size: int = 5120
    rope_theta: float = 10000.0


class Attention(nn.Module):
    """Multi-head attention with RoPE, QK-norm (RMSNorm on queries and keys)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.c_q = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.c_k = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.c_v = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        half_d = self.head_dim // 2
        self._rope_freqs = -mx.exp(
            mx.arange(0.0, half_d, dtype=mx.float32) * (math.log(args.rope_theta) / half_d)
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.c_q(x)
        keys = self.c_k(x)
        values = self.c_v(x)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = _apply_rope(queries, offset, self._rope_freqs, self.head_dim)
        keys = _apply_rope(keys, offset, self._rope_freqs, self.head_dim)

        queries = _rms_norm(queries)
        keys = _rms_norm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.c_proj(output)


class MLP(nn.Module):
    """MLP with ReLU² activation (nanochat-specific)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.c_fc = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.c_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = relu_squared(x)
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm block with functional RMSNorm and residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.attn(_rms_norm(x), mask=mask, cache=cache)
        return h + self.mlp(_rms_norm(h))


class NanoChatModel(nn.Module):
    """NanoChat transformer backbone: embed, norm, layers, final norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.wte(inputs)
        h = _rms_norm(h)

        if cache is None:
            cache = [None] * len(self.h)  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.h, cache, strict=True):
            h = layer(h, mask=mask, cache=c)

        h = _rms_norm(h)
        return h


class Model(nn.Module):
    """NanoChat LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = NanoChatModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.transformer(input_ids, cache=cache)
        logits = self.lm_head(out)
        return _softcap(logits)

    @property
    def num_layers(self) -> int:
        return len(self.transformer.h)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.transformer.h]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights

    def parameters(self) -> dict[str, Any]:
        return dict(self.items())
