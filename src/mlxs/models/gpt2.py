"""GPT-2 model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Uses learned position embeddings, fused QKV attention, and GELU activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.kv import KVCache
from mlxs.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)


@dataclass
class ModelArgs(BaseModelArgs):
    """GPT-2 model configuration."""

    model_type: str = "gpt2"
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    layer_norm_epsilon: float = 1e-5
    vocab_size: int = 50257
    num_key_value_heads: int | None = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.n_head


class Attention(nn.Module):
    """Multi-head attention with fused QKV projection (no RoPE)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.c_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """GPT-2 MLP with approximate GELU."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.c_fc = nn.Linear(args.n_embd, 4 * args.n_embd)
        self.c_proj = nn.Linear(4 * args.n_embd, args.n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu_approx(self.c_fc(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.ln_1 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.attn(self.ln_1(x), mask, cache)
        return h + self.mlp(self.ln_2(h))


class GPT2Model(nn.Module):
    """GPT-2 transformer backbone with learned position embeddings."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.n_layer = args.n_layer
        self.wte = nn.Embedding(args.vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.n_positions, args.n_embd)
        self.h = [TransformerBlock(args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        _, L = inputs.shape

        hidden_states = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.h)  # type: ignore[list-item]

        offset = 0
        if cache[0] is not None:
            offset = cache[0].offset

        offset = mx.array(offset)
        position_ids = mx.arange(L) + offset[..., None]
        hidden_states = hidden_states + self.wpe(position_ids)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.h, cache, strict=True):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):
    """GPT-2 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPT2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache)
        return self.model.wte.as_linear(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.h)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.h]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        new_weights: dict[str, Any] = {}
        for i in range(self.args.n_layer):
            # Remove causal mask buffers
            weights.pop(f"h.{i}.attn.bias", None)
            # Transpose Conv1D weights to Linear format
            for suffix in (
                "attn.c_attn.weight",
                "attn.c_proj.weight",
                "mlp.c_fc.weight",
                "mlp.c_proj.weight",
            ):
                key = f"h.{i}.{suffix}"
                if key in weights:
                    weights[key] = weights[key].transpose(1, 0)

        for key, value in weights.items():
            if not key.startswith("model."):
                new_weights[f"model.{key}"] = value
            else:
                new_weights[key] = value
        return new_weights

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.h
