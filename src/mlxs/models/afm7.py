"""AFM7 model: transformer with KV-reuse layers and optional 8-bit key/value quant.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Uses RMSNorm, SwiGLU MLP, RoPE, and a single concatenating KV cache for the
last attention layer whose keys/values are reused by subsequent KV-reuse layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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
    """AFM7 model configuration."""

    model_type: str = "afm7"
    vocab_size: int = 128256
    hidden_dim: int = 2048
    num_layers: int = 24
    num_kv_reuse_layers: int = 2
    num_heads: int = 16
    num_kv_heads: int = 16
    hidden_dim_scale_factor: float = 3.25
    rope_theta: float = 50000.0
    rms_norm_eps: float = 1e-5


@partial(mx.compile, shapeless=True)
def _fake_8bit_quant(x: mx.array, scale: mx.array) -> mx.array:
    """Fake 8-bit quantization for keys/values (AFM7-specific)."""
    dt = x.dtype
    x = x.astype(mx.float32)
    x = (x / scale).round()
    x = mx.clip(x, -128, 127)
    return (x * scale).astype(dt)


class _QKVLinear(nn.Module):
    """Single linear projecting to [q, k, v] then split (replaces FusedLinear for inference)."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        qkv_dim = n_heads * head_dim + 2 * n_kv_heads * head_dim
        self.linear = nn.Linear(dim, qkv_dim, bias=False)
        self._q_size = n_heads * head_dim
        self._k_size = n_kv_heads * head_dim
        self._v_size = n_kv_heads * head_dim

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        out = self.linear(x)
        q = out[..., : self._q_size]
        k = out[..., self._q_size : self._q_size + self._k_size]
        v = out[..., self._q_size + self._k_size :]
        return q, k, v


class Attention(nn.Module):
    """Multi-head attention with Q/K norms, RoPE, and optional fake 8-bit quant on K/V."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_dim
        self.n_heads = n_heads = args.num_heads
        self.n_kv_heads = n_kv_heads = args.num_kv_heads
        self.head_dim = head_dim = dim // n_heads
        self.scale = head_dim**-0.5

        self.qkv_proj = _QKVLinear(dim, n_heads, n_kv_heads, self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=True,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.quant_key_scale = mx.array(1.0)
        self.quant_value_scale = mx.array(1.0)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.qkv_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.q_norm(self.rope(queries, offset=cache.offset))
            keys = self.k_norm(self.rope(keys, offset=cache.offset))
            keys = _fake_8bit_quant(keys, self.quant_key_scale)
            values = _fake_8bit_quant(values, self.quant_value_scale)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.q_norm(self.rope(queries))
            keys = self.k_norm(self.rope(keys))
            keys = _fake_8bit_quant(keys, self.quant_key_scale)
            values = _fake_8bit_quant(values, self.quant_value_scale)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class KVReuseAttention(nn.Module):
    """Attention that reuses keys/values from the last transformer layer (no cache)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_dim
        self.n_heads = n_heads = args.num_heads
        self.head_dim = head_dim = dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=True,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        _, _, S, _ = keys.shape

        queries = self.q_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        queries = self.q_norm(self.rope(queries, offset=S - L))

        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_dim
        hidden_dim = int(dim * args.hidden_dim_scale_factor)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(swiglu(g, u))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with self-attention and MLP."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

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


class KVReuseTransformerBlock(nn.Module):
    """Block that reuses keys/values from the last attention layer."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = KVReuseAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), keys, values, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class AFMModel(nn.Module):
    """AFM7 backbone: embedding, transformer layers, KV-reuse layers, output norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.layers = [
            TransformerBlock(args) for _ in range(args.num_layers - args.num_kv_reuse_layers)
        ]
        self.kv_reuse_layers = [
            KVReuseTransformerBlock(args) for _ in range(args.num_kv_reuse_layers)
        ]
        self.output_norm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embedding(inputs)
        if cache is None:
            cache = [KVCache() for _ in range(len(self.layers))]

        mask = create_attention_mask(h, cache[-1])
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)

        last_cache = cache[-1]
        keys = last_cache.keys
        values = last_cache.values
        if keys is not None and values is not None:
            for layer in self.kv_reuse_layers:
                h = layer(h, keys, values, mask)

        return self.output_norm(h)


class Model(nn.Module):
    """AFM7 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AFMModel(args)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache)
        return self.model.embedding.as_linear(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers) + len(self.model.kv_reuse_layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(len(self.model.layers))]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights
