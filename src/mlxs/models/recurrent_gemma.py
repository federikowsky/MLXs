"""Recurrent Gemma (Griffin) — port from mlx_lm, ModelProtocol-compliant.

Hybrid recurrent + local attention: each layer is either a RecurrentBlock
(ArraysCache for conv + RG-LRU state) or LocalAttentionBlock (RotatingKVCache).
Mask is built from the first attention layer's cache and window_size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.norms import GemmaRMSNorm
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Args for Recurrent Gemma; from_dict aligned to config.json (mlx_lm)."""

    model_type: str
    attention_bias: bool
    conv1d_width: int
    hidden_size: int
    intermediate_size: int
    logits_soft_cap: float
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    attention_window_size: int
    vocab_size: int
    embeddings_scale_by_sqrt_dim: bool = True
    block_types: list[str] | None = None
    _block_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.block_types is None:
            self.block_types = self._block_types


def _rnn_scan(
    x: mx.array,
    a: mx.array,
    h0: mx.array | None,
) -> tuple[mx.array, mx.array]:
    """Linear recurrence: h_t = a_t * h_{t-1} + x_t. Returns (y, last_h)."""
    assert x.ndim == 3
    assert a.shape == x.shape[-a.ndim :]
    assert a.dtype == x.dtype

    if x.shape[1] == 1:
        if h0 is None:
            return x, x[:, 0]
        y = a * h0[:, None] + x
        return y, y[:, -1]

    B, _, D = x.shape
    h_t = h0 if h0 is not None else mx.zeros((B, D), dtype=x.dtype)
    y = mx.zeros_like(x)
    for t in range(x.shape[1]):
        h_t = a[:, t] * h_t + x[:, t]
        y[:, t] = h_t
    return y, h_t


class Conv1d(nn.Module):
    """Depthwise 1D conv; cache holds last (K-1) input frames for causal padding."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.weight = mx.zeros((channels, kernel_size, 1))
        self.bias = mx.zeros((channels,))

    def __call__(
        self,
        x: mx.array,
        cache: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        groups, K, _ = self.weight.shape
        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)
        else:
            x = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        y = mx.conv_general(x, self.weight, groups=groups)
        y = y + self.bias
        return y, x[:, -K + 1 :, :]


class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit (RG-LRU). State in cache[1]."""

    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.recurrent_param = mx.zeros((width,))
        self.input_gate_weight = mx.zeros((num_heads, self.head_dim, self.head_dim))
        self.input_gate_bias = mx.zeros((num_heads, self.head_dim))
        self.recurrent_gate_weight = mx.zeros((num_heads, self.head_dim, self.head_dim))
        self.recurrent_gate_bias = mx.zeros((num_heads, self.head_dim))

    def __call__(
        self,
        x: mx.array,
        cache: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        B, L, _ = x.shape

        def apply_block_linear(h: mx.array, w: mx.array, b: mx.array) -> mx.array:
            h = h.reshape((B, L, self.num_heads, self.head_dim))
            h = (h.swapaxes(1, 2) @ w).swapaxes(1, 2) + b
            return mx.sigmoid(h.flatten(2, 3))

        gate_x = apply_block_linear(x, self.input_gate_weight, self.input_gate_bias)
        gate_a = apply_block_linear(x, self.recurrent_gate_weight, self.recurrent_gate_bias)
        log_a = -8.0 * gate_a * nn.softplus(self.recurrent_param)
        a = mx.exp(log_a)
        a_square = mx.exp(2 * log_a)
        gated_x = x * gate_x
        multiplier = mx.sqrt(1 - a_square)
        if cache is None:
            multiplier = mx.concatenate(
                [mx.ones((B, 1, self.width), dtype=multiplier.dtype), multiplier[:, 1:, :]],
                axis=1,
            )
        normalized_x = gated_x * multiplier.astype(x.dtype)
        y, last_h = _rnn_scan(x=normalized_x, a=a, h0=cache)
        return y, last_h


class RecurrentBlock(nn.Module):
    """Temporal block: linear_y -> GELU, linear_x -> conv -> RG-LRU, then y * x -> linear_out."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
    ) -> None:
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.lru_width = lru_width or width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.linear_y = nn.Linear(width, self.lru_width)
        self.linear_x = nn.Linear(width, self.lru_width)
        self.linear_out = nn.Linear(self.lru_width, width)
        self.conv_1d = Conv1d(channels=self.lru_width, kernel_size=conv1d_temporal_width)
        self.rg_lru = RGLRU(width=self.lru_width, num_heads=num_heads)

    def __call__(
        self,
        x: mx.array,
        cache: list[Any] | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        y = self.linear_y(x)
        y = nn.gelu_approx(y)
        x = self.linear_x(x)
        if cache is None:
            cache = [None, None]
        x, cache[0] = self.conv_1d(x, cache=cache[0])
        x, cache[1] = self.rg_lru(x, cache=cache[1])
        x = x * y
        x = self.linear_out(x)
        return x


class LocalAttentionBlock(nn.Module):
    """Local attention with RoPE and RotatingKVCache (window_size)."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        window_size: int,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = width // num_heads
        self.scale = (self.head_dim) ** (-0.5)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, self.head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=True)
        self.rope = initialize_rope(
            self.head_dim // 2,
            base=rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        cache: RotatingKVCache | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, 1, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, 1, -1).transpose(0, 2, 1, 3)
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
        return self.o_proj(output)


class MLPBlock(nn.Module):
    """MLP: gate_proj -> GELU_approx * up_proj -> down_proj."""

    def __init__(self, width: int, expanded_width: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(width, expanded_width // 2)
        self.gate_proj = nn.Linear(width, expanded_width // 2)
        self.down_proj = nn.Linear(expanded_width // 2, width)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        x = self.up_proj(x)
        return self.down_proj(nn.gelu_approx(gate) * x)


class ResidualBlock(nn.Module):
    """One residual block: temporal (recurrent or attention) + channel MLP."""

    def __init__(
        self,
        width: int,
        mlp_expanded_width: int,
        num_heads: int,
        attention_window_size: int,
        temporal_block_type: str,
        rope_theta: float = 10000.0,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
    ) -> None:
        super().__init__()
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.num_heads = num_heads
        self.attention_window_size = attention_window_size
        self.temporal_block_type = temporal_block_type
        self.lru_width = lru_width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.temporal_pre_norm = GemmaRMSNorm(width, eps=1e-5)
        if temporal_block_type == "recurrent":
            self.temporal_block = RecurrentBlock(
                width=width,
                num_heads=num_heads,
                lru_width=lru_width,
                conv1d_temporal_width=conv1d_temporal_width,
            )
        else:
            self.temporal_block = LocalAttentionBlock(
                width=width,
                num_heads=num_heads,
                window_size=attention_window_size,
                rope_theta=rope_theta,
            )
        self.channel_pre_norm = GemmaRMSNorm(width, eps=1e-5)
        self.mlp_block = MLPBlock(width=width, expanded_width=mlp_expanded_width)

    def __call__(
        self,
        x: mx.array,
        cache: ArraysCache | RotatingKVCache | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        raw_x = x
        inputs_normalized = self.temporal_pre_norm(raw_x)
        layer_cache: Any = [cache[0], cache[1]] if isinstance(cache, ArraysCache) else cache
        x = self.temporal_block(inputs_normalized, cache=layer_cache, mask=mask)
        if isinstance(cache, ArraysCache):
            cache[0], cache[1] = layer_cache[0], layer_cache[1]
        residual = x + raw_x
        x = self.channel_pre_norm(residual)
        x = self.mlp_block(x)
        return x + residual


class Griffin(nn.Module):
    """Recurrent Gemma backbone: embed -> alternating recurrent/attention blocks -> norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.scale_by_sqrt_dim = args.embeddings_scale_by_sqrt_dim
        block_types = args.block_types or ["recurrent", "attention"]
        self.layers = [
            ResidualBlock(
                width=args.hidden_size,
                mlp_expanded_width=args.intermediate_size,
                num_heads=args.num_attention_heads,
                attention_window_size=args.attention_window_size,
                temporal_block_type=block_types[i % len(block_types)],
                rope_theta=args.rope_theta,
                lru_width=None,
                conv1d_temporal_width=args.conv1d_width,
            )
            for i in range(args.num_hidden_layers)
        ]
        self.final_norm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.window_size = args.attention_window_size
        self._block_types = block_types
        self._swa_idx: int | None = next(
            (i for i, bt in enumerate(block_types) if bt == "attention"), None
        )

    def __call__(
        self,
        tokens: mx.array,
        cache: list[ArraysCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        x = self.embed_tokens(tokens)
        if self.scale_by_sqrt_dim:
            x = x * math.sqrt(x.shape[-1])
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        attn_cache = cache[self._swa_idx] if self._swa_idx is not None and cache else None
        mask = create_attention_mask(x, attn_cache, window_size=self.window_size)
        for i, block in enumerate(self.layers):
            x = block(x, cache=cache[i], mask=mask)
        return self.final_norm(x)


class Model(nn.Module):
    """Recurrent Gemma LM head wrapper — satisfies ModelProtocol (AC17)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Griffin(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[ArraysCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        logits = self.model(input_ids, cache=cache)
        if hasattr(self, "lm_head") and getattr(self, "lm_head", None) is not None:
            logits = self.lm_head(logits)
        else:
            logits = self.model.embed_tokens.as_linear(logits)
        c = self.args.logits_soft_cap
        if c:
            logits = mx.tanh(logits / c) * c
        return logits

    def make_cache(self) -> list[ArraysCache | RotatingKVCache]:
        block_types = self.args.block_types or ["recurrent", "attention"]
        out: list[ArraysCache | RotatingKVCache] = []
        for i in range(self.args.num_hidden_layers):
            bt = block_types[i % len(block_types)]
            if bt == "recurrent":
                out.append(ArraysCache(size=2))
            else:
                out.append(
                    RotatingKVCache(
                        max_size=self.args.attention_window_size,
                        keep=0,
                    )
                )
        return out

    @property
    def num_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        for k, v in list(weights.items()):
            if "conv_1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        if "lm_head.weight" not in weights and hasattr(self, "lm_head"):
            delattr(self, "lm_head")
        return weights
