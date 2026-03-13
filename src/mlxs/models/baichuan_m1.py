"""Baichuan M1 — hybrid global + sliding-window attention, conv on K/V (§7).

Port from mlx_lm. Per-layer: optional conv state (ArraysCache) + KVCache or
RotatingKVCache. RMSNorm, SwiGLU MLP, RoPE. Implements ModelProtocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.arrays import ArraysCache
from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.cache_list import CacheList
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Baichuan M1 config; from_dict aligned to config.json (mlx_lm)."""

    model_type: str = "baichuan_m1"
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    rope_theta: float = 10000.0
    sliding_window: int = 4096
    sliding_window_layers: list[int] | None = None
    conv_window: int = 2
    rms_norm_eps: float = 1e-6
    num_swa_attention_heads: int | None = None
    num_swa_key_value_heads: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.sliding_window_layers is None:
            self.sliding_window_layers = []


def _custom_convolution(
    u: mx.array,
    weights: mx.array,
    state: mx.array | None,
    conv_window: int,
) -> tuple[mx.array, mx.array]:
    """Causal conv on last dimension: u_prev * w0 + u * w1; state = last step."""
    B, H, L, D = u.shape
    weights = weights.reshape((1, H, conv_window, 1, 1))
    w0 = weights[:, :, 0]
    w1 = weights[:, :, 1]
    if state is None:
        state = mx.zeros((B, H, 1, D), u.dtype)
    u_prev = mx.concatenate([state, u[:, :, :-1]], axis=2) if L > 1 else state
    out = u_prev * w0 + u * w1
    return out, u[:, :, -1:, :]


class Attention(nn.Module):
    """Multi-head attention with optional SWA, RoPE, and K/V convolution."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.is_swa = layer_idx in args.sliding_window_layers
        self.n_heads = (
            args.num_swa_attention_heads
            if self.is_swa and args.num_swa_attention_heads is not None
            else args.num_attention_heads
        )
        self.n_kv_heads = (
            args.num_swa_key_value_heads
            if self.is_swa and args.num_swa_key_value_heads is not None
            else args.num_key_value_heads
        )
        self.head_dim = args.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        assert args.conv_window == 2
        self.conv_window = args.conv_window

        in_dim = args.hidden_size
        qkv_out = self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.head_dim
        self.W_pack = nn.Linear(in_dim, qkv_out, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=None,
        )
        self.conv_k = mx.zeros((1, 1, self.n_kv_heads, 1, self.conv_window))
        self.conv_v = mx.zeros((1, 1, self.n_kv_heads, 1, self.conv_window))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: CacheList | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        proj = self.W_pack(x)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = mx.split(proj, (q_dim, q_dim + kv_dim), axis=-1)

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        conv_cache = cache[0] if cache is not None else None
        kv_cache = cache[1] if cache is not None else None

        offset = kv_cache.offset if kv_cache is not None else 0
        last_k = conv_cache[0] if conv_cache is not None else None
        last_v = conv_cache[1] if conv_cache is not None else None

        k_init, v_init = k, v
        k, _ = _custom_convolution(k, self.conv_k, last_k, self.conv_window)
        v, new_v = _custom_convolution(v, self.conv_v, last_v, self.conv_window)
        del new_v  # store k_init[:,:,-1:,:] and v_init[:,:,-1:,:] below

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
        if conv_cache is not None and L > 0:
            conv_cache[0] = k_init[:, :, -1:, :]
            conv_cache[1] = v_init[:, :, -1:, :]

        out = scaled_dot_product_attention(q, k, v, cache=kv_cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    """Pre-norm block: attention + MLP with residuals."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: CacheList | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class BaichuanModel(nn.Module):
    """Transformer backbone with global and sliding-window layers."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.sliding_window = args.sliding_window
        self.swa_layers = set(args.sliding_window_layers)

        self._first_global_idx: int | None = None
        self._first_swa_idx: int | None = None
        for i in range(args.num_hidden_layers):
            if i in self.swa_layers:
                if self._first_swa_idx is None:
                    self._first_swa_idx = i
            else:
                if self._first_global_idx is None:
                    self._first_global_idx = i

    def __call__(
        self,
        inputs: mx.array,
        cache: list[CacheList] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        c_global = cache[self._first_global_idx] if self._first_global_idx is not None else None
        c_swa = cache[self._first_swa_idx] if self._first_swa_idx is not None else None
        global_mask = create_attention_mask(h, c_global)
        swa_mask = create_attention_mask(h, c_swa, window_size=self.sliding_window)

        for i, (layer, c) in enumerate(zip(self.layers, cache, strict=True)):
            mask = swa_mask if i in self.swa_layers else global_mask
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    """Baichuan M1 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BaichuanModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[CacheList] | None = None,
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

    def make_cache(self) -> list[CacheList]:
        out: list[CacheList] = []
        for i in range(self.args.num_hidden_layers):
            conv_cache = ArraysCache(size=2)
            if i in self.args.sliding_window_layers:
                kv_cache: KVCache | RotatingKVCache = RotatingKVCache(
                    max_size=self.args.sliding_window, keep=0
                )
            else:
                kv_cache = KVCache()
            out.append(CacheList(conv_cache, kv_cache))
        return out

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if "lm_head.scales" not in weights and "lm_head.weight" in weights:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            w = w.astype(mx.float32)
            norm = mx.linalg.norm(w, axis=-1, keepdims=True)
            w = (w / (norm + 1e-7)).astype(dtype)
            weights["lm_head.weight"] = w
        return weights
