"""Phi-3 Small model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Uses LayerNorm, GeGELU MLP, MuP scaling, optional block-sparse attention,
RoPE, and tied embeddings with dummy token masking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Phi-3 Small model configuration."""

    model_type: str = "phi3small"
    hidden_size: int = 1024
    dense_attention_every_n_layers: int = 1
    ff_intermediate_size: int = 2048
    gegelu_limit: float = 256.0
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    layer_norm_epsilon: float = 1e-5
    vocab_size: int = 100352
    num_key_value_heads: int = 16
    mup_attn_multiplier: float = 1.0
    mup_use_scaling: bool = True
    mup_embedding_multiplier: float = 10.0
    mup_width_multiplier: float = 8.0
    rope_embedding_base: float = 1000000.0
    rope_position_scale: float = 1.0
    blocksparse_block_size: int = 64
    blocksparse_num_local_blocks: int = 16
    blocksparse_vert_stride: int = 8

    def __post_init__(self) -> None:
        if self.blocksparse_block_size not in (32, 64):
            raise ValueError(
                f"Unsupported blocksparse_block_size {self.blocksparse_block_size}; "
                "must be 32 or 64"
            )


def _gegelu_impl(a_gelu: mx.array, a_linear: mx.array, limit: float) -> mx.array:
    a_gelu = mx.where(
        mx.isinf(a_gelu),
        a_gelu,
        mx.clip(a_gelu, a_min=None, a_max=limit),
    )
    a_linear = mx.where(
        mx.isinf(a_linear),
        a_linear,
        mx.clip(a_linear, a_min=-limit, a_max=limit),
    )
    out_gelu = a_gelu * mx.sigmoid(1.702 * a_gelu)
    return out_gelu * (a_linear + 1.0)


def _gegelu(x: mx.array, limit: float) -> mx.array:
    a_gelu, a_linear = x[..., ::2], x[..., 1::2]
    return _gegelu_impl(a_gelu, a_linear, limit)


class Attention(nn.Module):
    """Multi-head attention with optional block-sparse pattern and MuP scale."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.n_q_per_kv = n_heads // n_kv_heads
        self.head_dim = head_dim = args.hidden_size // n_heads

        self.query_key_value = nn.Linear(dim, (self.n_heads + 2 * self.n_kv_heads) * head_dim)
        self.dense = nn.Linear(dim, dim)

        if args.mup_use_scaling:
            norm_factor = head_dim / args.mup_attn_multiplier
        else:
            norm_factor = math.sqrt(float(head_dim))
        self.scale = 1.0 / norm_factor

        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=args.rope_embedding_base,
            scale=args.rope_position_scale,
        )

        if layer_idx % args.dense_attention_every_n_layers == 0:
            self.block_sparse = True
            self.blocksparse_block_size = args.blocksparse_block_size
            self.blocksparse_num_local_blocks = args.blocksparse_num_local_blocks
            self.blocksparse_vert_stride = args.blocksparse_vert_stride
        else:
            self.block_sparse = False
            self.blocksparse_block_size = 0
            self.blocksparse_num_local_blocks = 0
            self.blocksparse_vert_stride = 0

    def _block_sparse_mask(self, q_len: int, kv_len: int) -> tuple[mx.array, mx.array]:
        vert_stride = self.blocksparse_vert_stride
        local_blocks = self.blocksparse_num_local_blocks
        block_size = self.blocksparse_block_size
        n_heads = self.n_heads

        kv_blocks = (kv_len + block_size - 1) // block_size
        q_blocks = (q_len + block_size - 1) // block_size
        q_pos = mx.arange(kv_blocks - q_blocks, kv_blocks)[None, :, None]
        k_pos = mx.arange(kv_blocks)[None, None]

        mask_vert_strided = (
            mx.arange(kv_blocks)[None, :] + mx.arange(1, n_heads + 1)[:, None]
        ) % vert_stride
        mask_vert_strided = (mask_vert_strided == 0)[:, None, :]

        block_mask = (q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)
        block_mask = block_mask.reshape(self.n_kv_heads, self.n_q_per_kv, *block_mask.shape[-2:])
        dense_mask = mx.repeat(mx.repeat(block_mask, block_size, axis=-1), block_size, axis=-2)
        return block_mask, dense_mask[..., -q_len:, :kv_len]

    def _block_sparse_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array | str | None,
    ) -> mx.array:
        queries = self.scale * queries
        B = queries.shape[0]
        L = queries.shape[2]
        queries = mx.reshape(queries, (B, self.n_kv_heads, self.n_q_per_kv, L, -1))
        keys = mx.expand_dims(keys, 2)
        values = mx.expand_dims(values, 2)

        _, dense_mask = self._block_sparse_mask(L, keys.shape[-2])
        scores = queries @ mx.swapaxes(keys, -1, -2)

        if mask is not None:
            if hasattr(mask, "dtype") and mask.dtype == mx.bool_:
                scores = scores + mx.where(
                    mask,
                    mx.array(0, scores.dtype),
                    mx.array(-float("inf"), scores.dtype),
                )
            else:
                scores = scores + mask
        scores = scores + mx.where(
            dense_mask,
            mx.array(0, scores.dtype),
            mx.array(-float("inf"), scores.dtype),
        )
        scores = mx.softmax(scores, axis=-1, precise=True)

        output = scores @ values
        return mx.reshape(output, (B, self.n_heads, L, -1))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, -1, self.n_q_per_kv + 2, self.head_dim)
        queries = qkv[..., :-2, :].flatten(-3, -2)
        keys = qkv[..., -2, :]
        values = qkv[..., -1, :]

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if self.block_sparse:
            output = self._block_sparse_attention(queries, keys, values, mask)
        else:
            output = scaled_dot_product_attention(queries, keys, values, cache, self.scale, mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class MLP(nn.Module):
    """MLP with GeGELU activation and up/down projections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.ff_intermediate_size
        self.gegelu_limit = args.gegelu_limit
        self.up_proj = nn.Linear(dim, 2 * hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.up_proj(x)
        return self.down_proj(_gegelu(x, self.gegelu_limit))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with LayerNorm and residual connections."""

    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)

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


class Phi3SmallModel(nn.Module):
    """Phi-3 Small transformer backbone with MuP embedding scaling."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.mup_embedding_multiplier = args.mup_embedding_multiplier
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.final_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if self.mup_embedding_multiplier != 0.0:
            h = self.mup_embedding_multiplier * h

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]
        mask = create_attention_mask(h, cache[0], return_array=True)
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)
        return self.final_layernorm(h)


# Dummy token IDs for phi3small (tied output): mask logits to -inf.
_DUMMY_TOKEN_IDS: list[int] = [
    100256,
    100258,
    100259,
    100260,
    100264,
    100265,
    *range(100267, 100352),
]


class Model(nn.Module):
    """Phi-3 Small LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Phi3SmallModel(args)
        self.mup_width_multiplier = args.mup_width_multiplier
        self._dummy_tokenizer_ids = mx.array(_DUMMY_TOKEN_IDS)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        if self.mup_width_multiplier != 0.0:
            out = out / self.mup_width_multiplier
        # Mask dummy token logits to -inf (only indices within vocab range)
        V = out.shape[-1]
        ids_to_mask = [i for i in _DUMMY_TOKEN_IDS if i < V]
        if ids_to_mask:
            keep = mx.ones((V,), dtype=mx.bool_).at[mx.array(ids_to_mask)].set(False)
            out = mx.where(
                keep[None, None, :],
                out,
                mx.array(-float("inf"), out.dtype),
            )
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
        return {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.layers
