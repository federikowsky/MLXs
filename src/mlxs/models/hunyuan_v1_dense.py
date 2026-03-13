"""Hunyuan V1 Dense model — ModelProtocol.

Dense decoder: GQA, RoPE (optional NTK alpha scaling), SwiGLU MLP, RMSNorm,
optional QK-norm. Port from mlx_lm.
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
from mlxs.models.base import BaseModelArgs

# ----- Model args -----


@dataclass
class ModelArgs(BaseModelArgs):
    """Hunyuan V1 Dense config (config.json)."""

    model_type: str = "hunyuan_v1_dense"
    vocab_size: int = 152064
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    attention_bias: bool = False
    use_qk_norm: bool = True
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if self.rope_scaling is not None:
            required = {"alpha", "factor", "type"}
            if not required.issubset(self.rope_scaling):
                raise ValueError(f"rope_scaling must contain {required}")


# ----- RoPE (model-specific: NTK alpha, fixed freqs) -----


class DynamicNTKAlphaRoPE(nn.Module):
    """RoPE with NTK alpha scaling: base = base * alpha^(dims/(dims-2)), fixed freqs."""

    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        scaling_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.dims = dims
        scaled_base = base * (scaling_alpha ** (dims / (dims - 2)))
        self._freqs = scaled_base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


# ----- Attention -----


class Attention(nn.Module):
    """GQA with optional QK-norm and Dynamic NTK Alpha RoPE."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim if args.head_dim is not None else dim // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)

        scaling_alpha = 1.0
        if args.rope_scaling and "alpha" in args.rope_scaling:
            scaling_alpha = float(args.rope_scaling["alpha"])
        self.rope = DynamicNTKAlphaRoPE(
            head_dim,
            base=args.rope_theta,
            scaling_alpha=scaling_alpha,
        )

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

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


# ----- MLP -----


class MLP(nn.Module):
    """SwiGLU MLP: gate_proj, up_proj -> swiglu -> down_proj."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


# ----- Decoder block -----


class TransformerBlock(nn.Module):
    """Pre-norm: norm -> attn -> residual, norm -> mlp -> residual."""

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
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


# ----- Backbone -----


class HunyuanV1DenseModel(nn.Module):
    """Embed + decoder layers + final RMSNorm."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | None] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


# ----- Model (ModelProtocol) -----


class Model(nn.Module):
    """Hunyuan V1 Dense LM — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HunyuanV1DenseModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        out = self.model(input_ids, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        assert self.lm_head is not None
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
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights
