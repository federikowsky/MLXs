"""Ministral3 model — port from mlx_lm, ModelProtocol-compliant.

Architecture: dense decoder with optional sliding-window layers, Llama 4-style
attention scaling, RoPE, SwiGLU MLP. Full-attention layers use KVCache;
sliding-attention layers use RotatingKVCache. Structure follows mlx_lm
models/ministral3.py (Attention, MLP, TransformerBlock, LanguageModel, Model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.kv import KVCache
from mlxs.cache.rotating import RotatingKVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


def _llama4_attn_scale(
    size: int,
    offset: int | mx.array,
    beta: float,
    max_position_embeddings: int,
    dtype: Any = None,
) -> mx.array:
    """Llama 4-style attention scaling: 1 + beta * log(1 + floor(pos / max_pos)).

    Matches mlx_lm ministral3 _get_llama_4_attn_scale. Returns shape (1, 1, size, 1)
    or (B, 1, size, 1) for broadcasting to (B, heads, L, 1).
    """
    dtype = dtype or mx.float32
    if isinstance(offset, mx.array) and offset.ndim > 0:
        offset_arr = offset[:, None]
    else:
        offset_arr = mx.array(offset, dtype=dtype)
    positions = mx.arange(size, dtype=dtype) + offset_arr
    scaling = 1.0 + beta * mx.log(
        1.0 + mx.floor(positions / max_position_embeddings).astype(dtype)
    )
    if scaling.ndim == 2:
        return scaling[:, None, :, None]
    return mx.reshape(scaling, (1, 1, -1, 1))


@dataclass
class ModelArgs(BaseModelArgs):
    """Ministral3 config; from_dict aligned to config.json (HF Ministral3Config).

    rope_parameters: dict with rope_theta, llama_4_scaling_beta,
    original_max_position_embeddings for RoPE and Llama 4 scale.
    """

    model_type: str = "ministral3"
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    intermediate_size: int = 5632
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    head_dim: int | None = None
    max_position_embeddings: int | None = None
    num_key_value_heads: int | None = None
    rope_parameters: dict[str, Any] | None = None
    tie_word_embeddings: bool = True
    layer_types: list[str] | None = None
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


class Attention(nn.Module):
    """Multi-head attention with GQA, RoPE, and optional Llama 4 query scaling.

    Queries are scaled by attn_scale (computed outside) before SDPA.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or dim // self.n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        rp = args.rope_parameters or {}
        rope_theta = float(rp.get("rope_theta", 10000.0))
        max_pos = args.max_position_embeddings or 32768
        self.rope = initialize_rope(
            self.head_dim,
            base=rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=max_pos,
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries * attn_scale
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """SwiGLU MLP (gate + up -> swiglu -> down)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm block: attn + MLP with residual. Optional sliding window (mask/cache)."""

    def __init__(self, args: ModelArgs, use_sliding: bool = False) -> None:
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | RotatingKVCache | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), attn_scale, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Ministral3Backbone(nn.Module):
    """Transformer backbone: embed, mixed full/sliding layers, norm.

    Builds fa_mask and swa_mask from first full_attention and first
    sliding_attention layer cache. Computes Llama 4 attn_scale once per forward.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types or ["full_attention"] * args.num_hidden_layers
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, use_sliding=(lt == "sliding_attention"))
            for lt in self.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._fa_idx = self.layer_types.index("full_attention")
        self._swa_idx: int | None = None
        for i, lt in enumerate(self.layer_types):
            if lt == "sliding_attention":
                self._swa_idx = i
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | RotatingKVCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        fa_mask = create_attention_mask(h, cache[self._fa_idx])
        swa_mask = None
        if self._swa_idx is not None and self.sliding_window is not None:
            swa_mask = create_attention_mask(
                h, cache[self._swa_idx], window_size=self.sliding_window
            )

        rp = self.args.rope_parameters or {}
        beta = float(rp.get("llama_4_scaling_beta", 0.1))
        orig_max = int(rp.get("original_max_position_embeddings", 4096))
        offset = cache[0].offset if cache and cache[0] is not None else 0
        attn_scale = _llama4_attn_scale(inputs.shape[1], offset, beta, orig_max, dtype=h.dtype)

        for layer, c in zip(self.layers, cache, strict=True):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, attn_scale, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """Ministral3 LM head wrapper — satisfies ModelProtocol."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Ministral3Backbone(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | RotatingKVCache] | None = None,
        mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(input_ids, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        out: list[KVCache | RotatingKVCache] = []
        sw = self.args.sliding_window
        for layer in self.model.layers:
            if layer.use_sliding and sw is not None:
                out.append(RotatingKVCache(max_size=sw, keep=0))
            else:
                out.append(KVCache())
        return out

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        weights = {k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        new_weights: dict[str, Any] = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights.get(wk)
                if weight is not None:
                    new_weights[wk] = weight * scale_inv
            elif "activation_scale" in k:
                continue
            elif k not in new_weights:
                new_weights[k] = v
        return new_weights
