"""Gemma 3 text model — port from mlx_lm (mlx_lm/models/gemma3_text.py).

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Registry: ``gemma3``; alias ``gemma3_text`` resolves to this module (same backbone).
Architecture: GQA, RoPE (local base for sliding layers, optional scaling for global),
Q/K head RMSNorm (1+weight), sliding-window + full attention pattern, clip_residual
for float16, separate lm_head (no tie by default). Cache mix: KVCache for global
layers, RotatingKVCache for sliding layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.norms import GemmaRMSNorm
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Gemma 3 text config; from_dict aligned to config.json (mlx_lm gemma3_text)."""

    model_type: str = "gemma3"
    hidden_size: int = 1152
    num_hidden_layers: int = 26
    intermediate_size: int = 6912
    num_attention_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 1
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 512
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 32768
    rope_scaling: dict[str, Any] | None = None


@partial(mx.compile, shapeless=True)
def _clip_residual(x: mx.array, y: mx.array) -> mx.array:
    """Clip residual for float16 stability (mlx_lm gemma3_text clip_residual)."""
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(mx.float16)


class Attention(nn.Module):
    """Multi-head attention with Q/K head norms and per-layer RoPE (local or global)."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.layer_idx = layer_idx
        self.scale = args.query_pre_attn_scalar**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        is_sliding = (layer_idx + 1) % args.sliding_window_pattern != 0
        if is_sliding:
            self.rope = nn.RoPE(
                self.head_dim,
                traditional=False,
                base=args.rope_local_base_freq,
            )
        else:
            self.rope = initialize_rope(
                dims=self.head_dim,
                base=args.rope_theta,
                traditional=False,
                max_position_embeddings=args.max_position_embeddings,
                scaling_config=args.rope_scaling,
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

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


class MLP(nn.Module):
    """Gated MLP with approximate GELU (same as Gemma 2)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm block with post-attn/post-ffn norms and clip_residual (mlx_lm)."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = _clip_residual(x, self.post_attention_layernorm(r))
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return _clip_residual(h, self.post_feedforward_layernorm(r))


class Gemma3Model(nn.Module):
    """Gemma 3 transformer backbone with sliding-window + full attention pattern."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.window_size = args.sliding_window
        self.sliding_window_pattern = args.sliding_window_pattern
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = GemmaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        global_mask = create_attention_mask(
            h,
            cache[self.sliding_window_pattern - 1],
            return_array=True,
        )
        if self.sliding_window_pattern > 1:
            sliding_window_mask = create_attention_mask(
                h,
                cache[0],
                window_size=self.window_size,
                return_array=True,
            )
        else:
            sliding_window_mask = None

        for i, (layer, c) in enumerate(zip(self.layers, cache, strict=True)):
            is_global = i % self.sliding_window_pattern == self.sliding_window_pattern - 1
            mask = global_mask if is_global else sliding_window_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """Gemma 3 LM head wrapper — satisfies ModelProtocol (AC17).

    Uses separate lm_head by default (tie_word_embeddings=False); no final
    logit soft-capping in base text model (mlx_lm gemma3_text).
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma3Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.tie_word_embeddings = False

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        caches: list[KVCache | RotatingKVCache] = []
        for i in range(self.args.num_hidden_layers):
            if i % self.args.sliding_window_pattern == self.args.sliding_window_pattern - 1:
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}
        if "lm_head.weight" not in out:
            self.tie_word_embeddings = True
        return out
