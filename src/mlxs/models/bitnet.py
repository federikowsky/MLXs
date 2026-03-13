"""BitNet model architecture — 1-bit linear layers, Llama-style layout.

Implements ModelProtocol. Compatible with mlx_lm-converted weights.
Uses BitLinear (ternary packed weights), RoPE, RMSNorm, sub-norms after attention/FFN.
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


def _make_bitlinear_kernel() -> mx.fast.metal_kernel | None:
    """Metal kernel for BitLinear matmul on packed ternary weights (4 values per uint8)."""
    if not mx.metal.is_available():
        return None
    source = """
    constexpr int M = 4;
    constexpr int BLOCK = 32;

    uint tid = thread_position_in_grid.y;
    uint in_offset = thread_position_in_grid.x;

    uint batch_idx = tid / (out_features / 4);
    uint row_idx = tid % (out_features / 4);

    float sum[4] = {0.0};

    for (uint i = in_offset * M; i < in_features; i += BLOCK * M) {
        float v[M];
        for (int j=0; j<M; j++) {
            v[j] = x[batch_idx * in_features + i + j];
        }

        for (int j=0; j<M; j++) {
            uint8_t w = packed_weights[row_idx * in_features + i + j];
            sum[0] += v[j] * ((w & 3) - 1);
            sum[1] += v[j] * (((w >> 2) & 3) - 1);
            sum[2] += v[j] * (((w >> 4) & 3) - 1);
            sum[3] += v[j] * (((w >> 6) & 3) - 1);
        }
    }

    for (int j=0; j<4; j++) {
        sum[j] = simd_sum(sum[j]);
    }

    if (in_offset == 0) {
        float scale = invert_weight_scales ? 1 / weight_scale[0] : weight_scale[0];
        for (int i=0; i<4; i++) {
            uint idx = batch_idx * out_features + row_idx + i * (out_features/4);
            out[idx] = static_cast<T>(sum[i] * scale);
        }
    }
    """
    return mx.fast.metal_kernel(
        name="bitlinear_matmul",
        input_names=["x", "packed_weights", "weight_scale"],
        output_names=["out"],
        source=source,
    )


_bitlinear_kernel = _make_bitlinear_kernel()


class BitLinear(nn.Module):
    """Linear layer with ternary packed weights (2 bits per weight, 4 per uint8)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        invert_weight_scales: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        packed_out = (out_features + 3) // 4
        self.weight = mx.zeros((packed_out, in_features), dtype=mx.uint8)
        self.invert_weight_scales = invert_weight_scales
        self.weight_scale = mx.array([1.0])
        self.bias = mx.zeros((out_features,)) if bias else None

    def _unpack_and_matmul(self, x: mx.array) -> mx.array:
        """Fallback when Metal kernel is unavailable: unpack ternary and matmul."""
        packed_rows, in_f = self.weight.shape
        out_f = self.out_features
        w = self.weight.astype(mx.float32)
        v0 = (w & 3) - 1
        v1 = ((w >> 2) & 3) - 1
        v2 = ((w >> 4) & 3) - 1
        v3 = ((w >> 6) & 3) - 1
        # (packed_rows, in_f, 4) -> (packed_rows*4, in_f) -> (out_f, in_f)
        unpacked = mx.concatenate(
            [v0[..., None], v1[..., None], v2[..., None], v3[..., None]], axis=-1
        )
        unpacked = unpacked.reshape(packed_rows * 4, in_f)[:out_f]
        scale = float(self.weight_scale)
        if self.invert_weight_scales:
            scale = 1.0 / scale
        out = (x.astype(mx.float32) @ unpacked.T * scale).astype(self.weight_scale.dtype)
        return out

    def __call__(self, x: mx.array) -> mx.array:
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])
        out_features = self.out_features
        dtype = self.weight_scale.dtype
        if x.dtype != dtype:
            x = x.astype(dtype)

        if _bitlinear_kernel is not None:
            total_batch_elements, in_features = x.shape
            out = _bitlinear_kernel(
                inputs=[x, self.weight, self.weight_scale],
                template=[
                    ("T", dtype),
                    ("invert_weight_scales", self.invert_weight_scales),
                    ("in_features", in_features),
                    ("out_features", out_features),
                ],
                grid=(32, total_batch_elements * out_features // 4, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(total_batch_elements, out_features)],
                output_dtypes=[dtype],
            )[0]
        else:
            out = self._unpack_and_matmul(x)

        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], out_features)
        if self.bias is not None:
            out = mx.add(out, self.bias)
        return out


@dataclass
class ModelArgs(BaseModelArgs):
    """BitNet model configuration."""

    model_type: str = "bitnet"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: dict[str, float | str] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    """Multi-head attention with BitLinear projections, RoPE, and output sub-norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or dim // self.n_heads
        self.scale = self.head_dim**-0.5
        bias = args.attention_bias

        self.q_proj = BitLinear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = BitLinear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = BitLinear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = BitLinear(self.n_heads * self.head_dim, dim, bias=bias)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        self.attn_sub_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        output = self.attn_sub_norm(output)
        return self.o_proj(output)


class MLP(nn.Module):
    """MLP with BitLinear and ReLU² gate, intermediate sub-norm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        mlp_bias = getattr(args, "mlp_bias", False)

        self.gate_proj = BitLinear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = BitLinear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = BitLinear(dim, hidden_dim, bias=mlp_bias)
        self.ffn_sub_norm = nn.RMSNorm(args.intermediate_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu2(self.gate_proj(x)) * self.up_proj(x)
        x = self.ffn_sub_norm(x)
        return self.down_proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

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


class BitNetModel(nn.Module):
    """BitNet transformer backbone (embed + layers + final norm)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """BitNet LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BitNetModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[KVCache] | None = None,
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

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        weights = {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights
