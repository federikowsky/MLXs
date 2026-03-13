"""OLMo model — port from mlx_lm (mlx_lm/models/olmo.py), ModelProtocol-compliant.

Dense decoder with LayerNorm (no affine), fused QKV projection, RoPE, SwiGLU MLP.
No dependency on ai2-olmo or hf_olmo.
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


class LayerNormNoAffine(nn.Module):
    """LayerNorm without learnable scale/bias (OLMo architecture)."""

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dims = dims
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(
            x, mx.ones((self.dims,)), mx.zeros((self.dims,)), self.eps
        )


@dataclass
class ModelArgs(BaseModelArgs):
    """OLMo config (config.json)."""

    model_type: str = "olmo"
    d_model: int = 1024
    n_layers: int = 16
    mlp_hidden_size: int | None = None
    n_heads: int = 16
    vocab_size: int = 50304
    embedding_size: int = 50304
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    mlp_ratio: int = 4
    weight_tying: bool = False

    def __post_init__(self) -> None:
        if self.mlp_hidden_size is None:
            self.mlp_hidden_size = self.mlp_ratio * self.d_model


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        dim = args.d_model
        head_dim = dim // self.n_heads

        self.ff_proj = nn.Linear(dim, args.mlp_hidden_size, bias=False)
        self.ff_out = nn.Linear(args.mlp_hidden_size // 2, dim, bias=False)
        self.att_norm = LayerNormNoAffine(dim)
        self.ff_norm = LayerNormNoAffine(dim)
        self.scale = head_dim**-0.5
        self.att_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=None,
            max_position_embeddings=None,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        qkv = self.att_proj(self.att_norm(x))
        queries, keys, values = mx.split(qkv, 3, axis=-1)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        r = self.attn_out(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
        h = x + r

        gate, up = mx.split(self.ff_proj(self.ff_norm(h)), 2, axis=-1)
        return h + self.ff_out(swiglu(up, gate))


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.weight_tying = args.weight_tying
        self.wte = nn.Embedding(args.embedding_size, args.d_model)
        self.blocks = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        if not self.weight_tying:
            self.ff_out = nn.Linear(args.d_model, args.embedding_size, bias=False)
        self.norm = LayerNormNoAffine(args.d_model)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array | tuple[mx.array, list[KVCache]]:
        h = self.wte(inputs)
        if cache is None:
            cache = [None] * len(self.blocks)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0])
        for block, c in zip(self.blocks, cache, strict=True):
            h = block(h, mask, cache=c)
        h = self.norm(h)
        if self.weight_tying:
            return self.wte.as_linear(h)
        return self.ff_out(h)


class OlmoModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.transformer = Transformer(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        out = self.transformer(inputs, cache)
        return out


class Model(nn.Module):
    """OLMo LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.model = OlmoModel(args)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        return self.model(input_ids, cache)

    @property
    def num_layers(self) -> int:
        return len(self.model.transformer.blocks)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.transformer.blocks]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights
