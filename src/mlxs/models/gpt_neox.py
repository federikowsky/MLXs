"""GPT-NeoX model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports parallel/sequential residual, partial RoPE, and approximate GELU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """GPT-NeoX model configuration."""

    model_type: str = "gpt_neox"
    max_position_embeddings: int = 2048
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    layer_norm_eps: float = 1e-5
    vocab_size: int = 50432
    rotary_emb_base: int = 10000
    rotary_pct: float = 0.25
    use_parallel_residual: bool = True
    num_key_value_heads: int | None = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    """Multi-head attention with partial RoPE and fused QKV."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.rope = initialize_rope(
            dims=int(self.head_dim * args.rotary_pct),
            base=float(args.rotary_emb_base),
            traditional=False,
        )

        self.query_key_value = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=True
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.query_key_value(x)
        new_qkv_shape = qkv.shape[:-1] + (self.num_attention_heads, 3 * self.head_dim)
        qkv = qkv.reshape(*new_qkv_shape)

        queries, keys, values = [
            t.transpose(0, 2, 1, 3) for t in qkv.split(3, -1)
        ]

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
        return self.dense(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """GPT-NeoX MLP with approximate GELU."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(args.hidden_size, 4 * args.hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * args.hidden_size, args.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense_4h_to_h(nn.gelu_approx(self.dense_h_to_4h(x)))


class TransformerBlock(nn.Module):
    """Transformer block with optional parallel residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.use_parallel_residual = args.use_parallel_residual
        self.attention = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        if self.use_parallel_residual:
            # Attention and MLP run in parallel on the same input.
            attn = self.attention(self.input_layernorm(x), mask, cache)
            ffn = self.mlp(self.post_attention_layernorm(x))
            return attn + ffn + x
        # Sequential: attention first, then MLP.
        h = x + self.attention(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class GPTNeoXModel(nn.Module):
    """GPT-NeoX transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embed_in = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_out = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.final_layer_norm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        hidden_states = self.embed_in(inputs)

        if cache is None:
            cache = [None] * len(self.h)  # type: ignore[list-item]

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.h, cache, strict=True):
            hidden_states = layer(hidden_states, mask, cache=c)

        out = self.final_layer_norm(hidden_states)
        return self.embed_out(out)


class Model(nn.Module):
    """GPT-NeoX LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPTNeoXModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        return self.model(inputs, cache)

    @property
    def num_layers(self) -> int:
        return len(self.model.h)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.h]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        ignore_suffixes = (
            ".attention.bias",
            ".attention.masked_bias",
            ".attention.rotary_emb.inv_freq",
        )
        new_weights: dict[str, Any] = {}
        for key, value in weights.items():
            if any(key.endswith(s) for s in ignore_suffixes):
                continue

            if not key.startswith("model."):
                key = f"model.{key}"

            key = key.replace(".gpt_neox.layers.", ".h.")
            key = key.replace(".gpt_neox.", ".")

            new_weights[key] = value
        return new_weights

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.h
