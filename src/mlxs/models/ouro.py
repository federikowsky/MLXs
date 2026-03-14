"""Ouro (LoopLM) model architecture for MLX.

Compatible with:
- ByteDance/Ouro-1.4B
- ByteDance/Ouro-2.6B
- ByteDance/Ouro-1.4B-Thinking
- ByteDance/Ouro-2.6B-Thinking
- mlx-community/Ouro-1.4B-4bit
- mlx-community/Ouro-2.6B-4bit
- mlx-community/Ouro-1.4B-Thinking-4bit
- mlx-community/Ouro-2.6B-Thinking-4bit

Design:
- decoder-only LoopLM / recurrent-depth transformer
- shared stack applied `total_ut_steps` times
- sandwich normalization (4 RMSNorms per block)
- RoPE attention, SwiGLU MLP
- one KV cache slot per (ut_step, layer): total slots = num_hidden_layers * total_ut_steps

Notes:
- Fixed-depth inference: early-exit gate is computed but not used to stop early.
- Quantized mlx-community checkpoints are dequantized in sanitize().
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

_ALLOWED_GROUP_SIZES = (32, 64)


@dataclass
class ModelArgs(BaseModelArgs):
    """Ouro model configuration."""

    model_type: str = "ouro"
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    intermediate_size: int = 5632
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 65536
    vocab_size: int = 49152
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    layer_types: list[str] | None = None
    total_ut_steps: int = 4
    early_exit_threshold: float = 1.0
    attention_dropout: float = 0.0
    sliding_window: int | None = None
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length must equal num_hidden_layers "
                f"({self.num_hidden_layers}), got {len(self.layer_types)}"
            )

        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim: "
                f"{self.hidden_size} != {self.num_attention_heads} * {self.head_dim}"
            )

        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("num_attention_heads and num_key_value_heads must be positive")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be divisible by "
                f"num_key_value_heads={self.num_key_value_heads}"
            )

        if self.total_ut_steps <= 0:
            raise ValueError(f"total_ut_steps must be positive, got {self.total_ut_steps}")

        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {self.max_position_embeddings}"
            )


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE.

    RoPE is applied inside attention using the cache offset. For standard
    monotonic causal generation, this is equivalent to computing position
    embeddings once at model level and passing them down.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape

        queries = (
            self.q_proj(x)
            .reshape(batch, seq_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )

        return self.o_proj(
            output.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.n_heads * self.head_dim)
        )


class MLP(nn.Module):
    """SwiGLU MLP (SiLU gate)."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        hidden = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    """Ouro decoder block with sandwich normalization."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        eps = args.rms_norm_eps
        self.self_attn = Attention(args)
        self.mlp = MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.input_layernorm_2 = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm_2 = nn.RMSNorm(args.hidden_size, eps=eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache)
        x = self.input_layernorm_2(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = self.post_attention_layernorm_2(x)
        x = residual + x
        return x


class OuroModel(nn.Module):
    """Ouro backbone: embeddings, looped layers, final norm, early-exit gate."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.early_exit_gate = nn.Linear(args.hidden_size, 1, bias=True)
        self.total_ut_steps = args.total_ut_steps

    def __call__(
        self,
        inputs: mx.array | None = None,
        cache: list[KVCache] | None = None,
        mask: mx.array | str | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if inputs is None and input_embeddings is None:
            raise ValueError("One of `inputs` or `input_embeddings` must be provided.")
        if inputs is not None and input_embeddings is not None:
            raise ValueError("Only one of `inputs` or `input_embeddings` may be provided.")

        n_layers = self.args.num_hidden_layers
        expected_cache_len = n_layers * self.total_ut_steps

        if cache is None:
            cache = [None] * expected_cache_len  # type: ignore[list-item]
        elif len(cache) != expected_cache_len:
            raise ValueError(
                "Ouro cache length must equal num_hidden_layers * total_ut_steps: "
                f"{expected_cache_len}, got {len(cache)}"
            )

        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        attn_mask = mask if mask is not None else create_attention_mask(h, cache[0])

        for current_ut in range(self.total_ut_steps):
            for layer_idx, layer in enumerate(self.layers):
                cache_idx = current_ut * n_layers + layer_idx
                h = layer(h, mask=attn_mask, cache=cache[cache_idx])
            h = self.norm(h)
            _ = self.early_exit_gate(h)

        return h


class Model(nn.Module):
    """Ouro LM wrapper."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OuroModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[KVCache] | None = None,
        mask: mx.array | str | None = None,
        input_embeddings: mx.array | None = None,
        **_kwargs: Any,
    ) -> mx.array:
        h = self.model(
            inputs=inputs,
            cache=cache,
            mask=mask,
            input_embeddings=input_embeddings,
        )
        return self.lm_head(h)

    @property
    def num_layers(self) -> int:
        return self.args.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.args.vocab_size

    def make_cache(self) -> list[KVCache]:
        n = self.args.num_hidden_layers * self.args.total_ut_steps
        return [KVCache() for _ in range(n)]

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        out = {
            k: v
            for k, v in weights.items()
            if "rotary_emb" not in k and "inv_freq" not in k
        }

        if self.args.tie_word_embeddings:
            out.pop("lm_head.weight", None)

        out = _dequantize_ouro_weights(
            out,
            hidden_size=self.args.hidden_size,
        )

        gate_key = "model.early_exit_gate.weight"
        if gate_key in out:
            w = out[gate_key]
            need = self.args.hidden_size

            if w.ndim != 2:
                raise ValueError(f"{gate_key}: expected 2D tensor, got shape {w.shape}")

            if w.shape == (need, 1):
                out[gate_key] = w.T
            elif w.shape == (1, need):
                pass
            else:
                raise ValueError(
                    f"{gate_key}: got shape {w.shape}; expected "
                    f"(1, {need}) or ({need}, 1). "
                    "This usually indicates wrong bits/group_size inference or a "
                    "checkpoint format mismatch."
                )

        gate_bias_key = "model.early_exit_gate.bias"
        if gate_bias_key in out:
            b = out[gate_bias_key]
            if b.ndim != 1 or b.shape[0] != 1:
                raise ValueError(
                    f"{gate_bias_key}: expected shape (1,), got {b.shape}"
                )

        return out


def _pack_uint8_to_uint32(w: mx.array) -> mx.array:
    """Reinterpret packed uint8 storage as uint32 for MLX dequantize."""
    import numpy as np

    w_np = np.asarray(w)
    if w_np.dtype != np.uint8:
        raise ValueError(f"Expected uint8 for repack, got {w_np.dtype}")

    if w_np.ndim < 1:
        raise ValueError(f"Expected at least 1D packed tensor, got shape {w_np.shape}")

    if w_np.shape[-1] % 4 != 0:
        raise ValueError(
            f"Packed uint8 tensor last dimension must be divisible by 4, got shape {w_np.shape}"
        )

    if not w_np.flags.c_contiguous:
        w_np = np.ascontiguousarray(w_np)

    packed = np.frombuffer(w_np.tobytes(), dtype=np.uint32)
    new_shape = (*w_np.shape[:-1], w_np.shape[-1] // 4)
    packed = packed.reshape(new_shape)
    return mx.array(packed)


def _infer_group_size_for_bits(
    key: str,
    packed_weight: mx.array,
    scales: mx.array,
    bits: int,
) -> int:
    if scales.size <= 0:
        raise ValueError(f"{key}: scales size must be positive, got {scales.size}")

    elements_per_u32 = 4 if bits == 8 else 8
    total_elements = packed_weight.size * elements_per_u32

    if total_elements % scales.size != 0:
        raise ValueError(
            f"{key}: total_elements={total_elements} not divisible by "
            f"num_groups={scales.size} (bits={bits})"
        )

    group_size = total_elements // scales.size
    if group_size not in _ALLOWED_GROUP_SIZES:
        raise ValueError(
            f"{key}: inferred group_size={group_size} not in allowed "
            f"{_ALLOWED_GROUP_SIZES} for bits={bits}"
        )
    return group_size


def _infer_early_exit_gate_bits_and_dequantize(
    key: str,
    packed_weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    hidden_size: int,
) -> mx.array:
    candidates: list[tuple[int, mx.array]] = []

    for bits in (4, 8):
        try:
            group_size = _infer_group_size_for_bits(key, packed_weight, scales, bits)
            dense = mx.dequantize(
                packed_weight,
                scales,
                biases,
                bits=bits,
                group_size=group_size,
            )
            if dense.ndim == 2 and dense.shape in ((1, hidden_size), (hidden_size, 1)):
                candidates.append((bits, dense))
        except Exception:
            continue

    if len(candidates) == 1:
        return candidates[0][1]

    if len(candidates) == 0:
        raise ValueError(
            f"{key}: could not infer whether checkpoint stores early_exit_gate "
            f"as 4-bit or 8-bit. No candidate produced shape "
            f"(1, {hidden_size}) or ({hidden_size}, 1)."
        )

    valid_bits = [bits for bits, _ in candidates]
    raise ValueError(
        f"{key}: ambiguous quantization format for early_exit_gate; multiple bit-widths "
        f"{valid_bits} produced plausible shapes."
    )


def _dequantize_ouro_weights(
    weights: dict[str, mx.array],
    hidden_size: int,
) -> dict[str, mx.array]:
    """Dequantize mlx-community affine quantized weights to dense tensors."""
    result: dict[str, mx.array] = {}

    for key in sorted(weights.keys()):
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if ".scales" in key or ".biases" in key:
            continue

        scales_key = key.replace(".weight", ".scales")
        biases_key = key.replace(".weight", ".biases")

        if scales_key not in weights or biases_key not in weights:
            result[key] = weights[key]
            continue

        w = weights[key]
        s = weights[scales_key]
        b = weights[biases_key]

        if s.size != b.size:
            raise ValueError(
                f"{key}: scales and biases must have matching sizes, got "
                f"{s.size} vs {b.size}"
            )

        if w.dtype == mx.uint8:
            w = _pack_uint8_to_uint32(w)
        elif w.dtype == mx.uint32:
            pass
        else:
            result[key] = w
            continue

        if key == "model.early_exit_gate.weight":
            result[key] = _infer_early_exit_gate_bits_and_dequantize(
                key, w, s, b, hidden_size
            )
            continue

        bits = 4
        group_size = _infer_group_size_for_bits(key, w, s, bits)
        result[key] = mx.dequantize(w, s, b, bits=bits, group_size=group_size)

    return result