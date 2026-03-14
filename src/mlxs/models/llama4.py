"""Llama 4 model — full (multimodal text tower) and text-only variant.

Port from mlx_lm models/llama4.py and llama4_text.py. Supports nested
text_config (full) and flat config (llama4_text). Chunked attention every
3 of 4 layers, optional MoE, QK norm, attn_temperature_tuning for dense layers.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlxs.cache.attention_mask import create_attention_mask
from mlxs.cache.chunked import ChunkedKVCache
from mlxs.cache.kv import KVCache
from mlxs.layers.activations import swiglu
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.layers.moe import SwitchGLU
from mlxs.layers.rope import initialize_rope
from mlxs.models.base import BaseModelArgs


@dataclass
class TextArgs(BaseModelArgs):
    """Text tower config — from text_config (llama4) or flat (llama4_text)."""

    model_type: str = "llama4"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 11008
    intermediate_size_mlp: int = 11008
    rms_norm_eps: float = 1e-6
    vocab_size: int = 128256
    rope_theta: float = 10000.0
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    use_qk_norm: bool = True
    # Chunked / MoE (full llama4)
    attention_chunk_size: int = 0
    interleave_moe_layer_step: int = 1
    num_local_experts: int = 0
    num_experts_per_tok: int = 1
    # Dense layer temperature (full llama4)
    attn_temperature_tuning: int = 0
    floor_scale: int = 8192
    attn_scale: float = 0.1
    # Text-only: per-layer use_rope (no_rope_layers[i] = use_rope for layer i)
    no_rope_layers: list[bool] | None = None
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> TextArgs:
        sig = inspect.signature(cls)
        return cls(**{k: v for k, v in params.items() if k in sig.parameters})


@dataclass
class ModelArgs(BaseModelArgs):
    """Llama 4 config: nested text_config (full) or flat (llama4_text)."""

    model_type: str = "llama4"
    text_config: dict[str, Any] | None = None
    _text_args: TextArgs | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        if "text_config" in params:
            text_args = TextArgs.from_dict(params["text_config"])
        else:
            text_args = TextArgs.from_dict(params)
        return cls(
            model_type=params.get("model_type", "llama4"),
            text_config=params.get("text_config"),
            _text_args=text_args,
        )

    @property
    def args(self) -> TextArgs:
        if self._text_args is not None:
            return self._text_args
        if self.text_config is not None:
            return TextArgs.from_dict(self.text_config)
        return TextArgs(model_type=self.model_type)


def _use_rope_for_layer(args: TextArgs, layer_idx: int) -> bool:
    if args.no_rope_layers is not None:
        return bool(args.no_rope_layers[layer_idx])
    return (layer_idx + 1) % 4 != 0


class Attention(nn.Module):
    """Multi-head attention with optional RoPE, QK norm, temperature tuning (mlx_lm Attention)."""

    def __init__(self, args: TextArgs, layer_idx: int) -> None:
        super().__init__()
        dim = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim or dim // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.use_rope = _use_rope_for_layer(args, layer_idx)
        self.use_qk_norm = args.use_qk_norm and self.use_rope
        self.attn_temperature_tuning = getattr(args, "attn_temperature_tuning", 0)
        self.floor_scale = getattr(args, "floor_scale", 8192)
        self.attn_scale = getattr(args, "attn_scale", 0.1)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        if self.use_rope:
            self.rope = initialize_rope(
                head_dim,
                base=args.rope_theta,
                traditional=True,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ChunkedKVCache | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        if self.use_rope:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if self.use_qk_norm:
            queries = mx.fast.rms_norm(queries, weight=None, eps=1e-6)
            keys = mx.fast.rms_norm(keys, weight=None, eps=1e-6)

        if self.attn_temperature_tuning and not self.use_rope:
            pos = mx.arange(offset + 1, offset + L + 1, dtype=mx.float32)
            attn_scales = mx.log(mx.floor(pos / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            attn_scales = attn_scales[:, None]
            queries = (queries * attn_scales).astype(queries.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    """SwiGLU MLP (mlx_lm MLP)."""

    def __init__(self, args: TextArgs, intermediate_size: int | None = None) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoE(nn.Module):
    """1 expert per token + shared expert (mlx_lm MoE, SwitchGLU experts)."""

    def __init__(self, args: TextArgs) -> None:
        super().__init__()
        assert args.num_experts_per_tok == 1, "Only 1 expert per token supported"
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_local_experts
        self.experts = SwitchGLU(args.hidden_size, args.intermediate_size, self.num_experts)
        self.router = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.shared_expert = MLP(args)

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.router(x)
        k = self.top_k
        indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(logits, indices, axis=-1)
        scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)
        out = self.experts(x * scores, indices).squeeze(2)
        return out + self.shared_expert(x)


class TransformerBlock(nn.Module):
    """Pre-norm block: attention + MLP or MoE (mlx_lm TransformerBlock)."""

    def __init__(self, args: TextArgs, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.is_moe_layer = (layer_idx % args.interleave_moe_layer_step) == (
            args.interleave_moe_layer_step - 1
        ) and args.num_local_experts > 0
        if self.is_moe_layer:
            self.feed_forward = MoE(args)
        else:
            self.feed_forward = MLP(args, args.intermediate_size_mlp)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | ChunkedKVCache | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.feed_forward(self.post_attention_layernorm(h))


class LlamaModel(nn.Module):
    """Text backbone: chunked (full) or full attention (text-only)."""

    def __init__(self, args: TextArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.attention_chunk_size = args.attention_chunk_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._dense_layer_idx = next(
            (i for i in range(len(self.layers)) if (i + 1) % 4 == 0),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache | ChunkedKVCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)  # type: ignore[list-item]

        if self.attention_chunk_size > 0:
            for idx, c in enumerate(cache):
                if c is not None and (idx + 1) % 4 != 0 and hasattr(c, "maybe_trim_front"):
                    c.maybe_trim_front()
            start = getattr(cache[0], "start_position", 0) if cache[0] is not None else 0
            offset = cache[0].offset if cache[0] is not None else 0
            end = offset + h.shape[1]
            linds = mx.arange(start, end)
            rinds = mx.arange(offset, end)[:, None]
            block_pos = mx.abs(
                (linds // self.attention_chunk_size) - (rinds // self.attention_chunk_size)
            )
            token_pos = linds <= rinds
            chunk_mask = (block_pos == 0) & token_pos
        else:
            offset = cache[0].offset if cache[0] is not None else 0
            chunk_mask = None

        global_mask = create_attention_mask(
            h,
            cache[self._dense_layer_idx] if self._dense_layer_idx is not None else cache[0],
        )

        for idx, (layer, c) in enumerate(zip(self.layers, cache, strict=True)):
            use_chunked = self.attention_chunk_size > 0 and (idx + 1) % 4 != 0
            mask = chunk_mask if use_chunked else global_mask
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """Llama 4 LM — ModelProtocol; full (nested text_config) or text-only (flat)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = args.args
        self.model = LlamaModel(text_args)
        self.lm_head = nn.Linear(text_args.hidden_size, text_args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[KVCache | ChunkedKVCache] | None = None,
        mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        out = self.model(input_ids, cache, input_embeddings=input_embeddings)
        return self.lm_head(out)

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def vocab_size(self) -> int:
        return self.args.args.vocab_size

    def make_cache(self) -> list[KVCache | ChunkedKVCache]:
        text_args = self.args.args
        chunk_size = text_args.attention_chunk_size
        caches: list[KVCache | ChunkedKVCache] = []
        for i in range(len(self.model.layers)):
            if chunk_size > 0 and (i + 1) % 4 != 0:
                caches.append(ChunkedKVCache(chunk_size))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        def to_remove(k: str) -> bool:
            return "vision_model" in k or "multi_modal_projector" in k

        weights = {k: v for k, v in weights.items() if not to_remove(k)}
        text_args = self.args.args
        for layer_idx in range(text_args.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer_idx}.feed_forward.experts"
            if f"{prefix}.gate_up_proj" in weights:
                v = weights.pop(f"{prefix}.gate_up_proj")
                gate_k = f"{prefix}.gate_proj.weight"
                up_k = f"{prefix}.up_proj.weight"
                gate_proj, up_proj = mx.split(v, 2, axis=-1)
                weights[gate_k] = mx.swapaxes(gate_proj, 1, 2)
                weights[up_k] = mx.swapaxes(up_proj, 1, 2)
            if f"{prefix}.down_proj" in weights:
                down_proj = weights.pop(f"{prefix}.down_proj")
                weights[f"{prefix}.down_proj.weight"] = mx.swapaxes(down_proj, 1, 2)
        return weights
