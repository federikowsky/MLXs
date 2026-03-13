"""OpenELM model architecture.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Supports per-layer variable head counts and FFN dimensions, QK normalization,
and optional weight-tied embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.models.base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """OpenELM model configuration."""

    model_type: str = "openelm"
    head_dim: int = 64
    num_transformer_layers: int = 16
    model_dim: int = 2048
    vocab_size: int = 32000
    ffn_dim_divisor: int = 256
    num_query_heads: list[int] | None = None
    num_kv_heads: list[int] | None = None
    ffn_multipliers: list[float] | None = None
    ffn_with_glu: bool = True
    normalize_qk_projections: bool = True
    share_input_output_layers: bool = True
    rms_norm_eps: float = 1e-6
    rope_freq_constant: float = 10000.0


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """Ensure value is divisible by the divisor.

    Taken from the original TensorFlow repo for MobileNet.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Attention(nn.Module):
    """Multi-head attention with per-layer head counts and optional QK norm."""

    def __init__(self, args: ModelArgs, layer_id: int) -> None:
        super().__init__()
        self.head_dim = head_dim = args.head_dim
        self.layer_id = layer_id
        self.model_dim = model_dim = args.model_dim

        self.n_heads = n_heads = args.num_query_heads[layer_id]
        self.n_kv_heads = n_kv_heads = args.num_kv_heads[layer_id]
        self.scale = head_dim**-0.5

        op_size = (n_heads + (n_kv_heads * 2)) * head_dim
        self.qkv_proj = nn.Linear(model_dim, op_size, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, model_dim, bias=False)

        self.normalize_qk_projections = args.normalize_qk_projections

        if self.normalize_qk_projections:
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_freq_constant)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(
            B, L, self.n_heads + (self.n_kv_heads * 2), self.head_dim
        ).transpose(0, 2, 1, 3)

        queries, keys, values = mx.split(
            qkv, [self.n_heads, self.n_heads + self.n_kv_heads], axis=1
        )

        if self.normalize_qk_projections:
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
        return self.out_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP with per-layer intermediate dimension."""

    def __init__(self, args: ModelArgs, layer_id: int) -> None:
        super().__init__()
        dim = args.model_dim
        ffn_multiplier = args.ffn_multipliers[layer_id]

        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * args.model_dim,
                divisor=args.ffn_dim_divisor,
            )
        )

        self.proj_1 = nn.Linear(dim, 2 * intermediate_dim, bias=False)
        self.proj_2 = nn.Linear(intermediate_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj_1(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.proj_2(swiglu(gate, x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with per-layer dimensions."""

    def __init__(self, args: ModelArgs, layer_id: int) -> None:
        super().__init__()
        dim = args.model_dim
        self.attn = Attention(args, layer_id=layer_id)
        self.ffn = MLP(args, layer_id=layer_id)
        self.ffn_norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.attn_norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        r = self.attn(self.attn_norm(x), mask, cache)
        h = x + r
        r = self.ffn(self.ffn_norm(h))
        return h + r


class OpenELMModel(nn.Module):
    """OpenELM transformer backbone."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_transformer_layers = args.num_transformer_layers
        assert self.vocab_size > 0
        self.token_embeddings = nn.Embedding(args.vocab_size, args.model_dim)
        self.layers = [
            TransformerBlock(args, layer_id=layer_id)
            for layer_id in range(self.num_transformer_layers)
        ]
        self.norm = nn.RMSNorm(args.model_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.token_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """OpenELM LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = OpenELMModel(args)
        if not args.share_input_output_layers:
            self.lm_head = nn.Linear(args.model_dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.transformer(inputs, cache)
        if self.args.share_input_output_layers:
            return self.transformer.token_embeddings.as_linear(out)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.transformer.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.transformer.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.transformer.layers
