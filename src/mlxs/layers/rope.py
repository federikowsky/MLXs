"""Shared RoPE utilities — all rotary position embedding variants.

Provides initialize_rope() factory and specialized RoPE classes:
- Default/linear: via nn.RoPE
- Llama3RoPE: per-dimension frequency scaling
- YarnRoPE: yarn/deepseek_yarn/telechat3-yarn scaling
- SuScaledRoPE: longrope-style with short/long factors

Compatible with mlx_lm rope_utils.py API.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _offset_nonzero(offset: int | mx.array) -> bool:
    if isinstance(offset, int):
        return offset != 0
    shape = getattr(offset, "shape", ())
    if len(shape) == 0:
        return bool(offset.item())
    return bool(mx.any(offset != 0).item())


def _offset_for_row(offset: int | mx.array, index: int) -> int | mx.array:
    if isinstance(offset, int):
        return offset
    shape = getattr(offset, "shape", ())
    if len(shape) == 0:
        return offset
    if shape[0] == 1:
        return offset
    return offset[index : index + 1]


def _apply_rope_safely(
    x: mx.array,
    offset: int | mx.array,
    apply_fn,
) -> mx.array:
    """Route the batched single-token decode case through row-wise RoPE.

    Real-model probes on the canonical Llama path showed that ``mx.fast.rope``
    can diverge across identical batch rows specifically for the grouped decode
    shape ``(B>1, H, 1, D)`` with a non-zero cache offset. Row-wise application
    preserves parity while keeping the batched fast path for prefill and other
    unaffected shapes.
    """
    if x.shape[0] <= 1 or x.shape[-2] != 1 or not _offset_nonzero(offset):
        return apply_fn(x, offset)

    return mx.concatenate(
        [
            apply_fn(x[index : index + 1], _offset_for_row(offset, index))
            for index in range(x.shape[0])
        ],
        axis=0,
    )


class DefaultRoPE(nn.Module):
    """Safe wrapper around ``nn.RoPE`` for grouped decode."""

    def __init__(
        self,
        dims: int,
        *,
        traditional: bool = False,
        base: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self._rope = nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    def __call__(self, x: mx.array, offset: int | mx.array = 0) -> mx.array:
        return _apply_rope_safely(x, offset, self._rope)


class SuScaledRoPE(nn.Module):
    """Su Scaled Rotary Embedding (longrope-style)."""

    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: list[float] | float = 1.0,
        long_factor: list[float] | float = 1.0,
        short_mscale: float | None = None,
        long_mscale: float | None = None,
    ) -> None:
        super().__init__()
        self.original_max_position_embeddings = original_max_position_embeddings
        self.dim = dims

        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._freqs = mx.array(long_factor, dtype=mx.float32) * freqs

        def default_scale(factor: float) -> float:
            return math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

        factor = max_position_embeddings / original_max_position_embeddings
        self._scale = long_mscale or (1.0 if factor <= 1.0 else default_scale(factor))

    def __call__(self, x: mx.array, offset: int | mx.array = 0) -> mx.array:
        x[..., : self.dim] = self._scale * x[..., : self.dim]
        return _apply_rope_safely(
            x,
            offset,
            lambda arr, off: mx.fast.rope(
                arr,
                self.dim,
                traditional=False,
                base=None,
                scale=1.0,
                offset=off,
                freqs=self._freqs,
            ),
        )


class Llama3RoPE(nn.Module):
    """Llama 3 per-dimension frequency scaling RoPE."""

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        if scaling_config is None:
            scaling_config = {}
        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def extra_repr(self) -> str:
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_position_embeddings={self.max_position_embeddings}"
        )

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return _apply_rope_safely(
            x,
            offset,
            lambda arr, off: mx.fast.rope(
                arr,
                self.dims,
                traditional=self.traditional,
                base=None,
                scale=1.0,
                offset=off,
                freqs=self._freqs,
            ),
        )


class DynamicNTKScalingRoPE(nn.Module):
    """RoPE with Dynamic NTK scaling (InternLM2-style)."""

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.original_base = base
        self.dims = dims
        self.traditional = traditional
        self.scale = scale

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seq_len = x.shape[1] + offset
        if seq_len > self.max_position_embeddings:
            base = self.original_base * (
                (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
            ) ** (self.dims / (self.dims - 2))
        else:
            base = self.original_base
        return _apply_rope_safely(
            x,
            offset,
            lambda arr, off: mx.fast.rope(
                arr,
                self.dims,
                traditional=self.traditional,
                base=base,
                scale=self.scale,
                offset=off,
            ),
        )


class YarnRoPE(nn.Module):
    """Yarn RoPE — supports yarn, deepseek_yarn, telechat3-yarn."""

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        attn_factor: float | None = None,
    ) -> None:
        super().__init__()

        if attn_factor is not None:
            mscale = float(attn_factor)

        def yarn_find_correction_dim(num_rotations: float) -> float:
            return (
                dims * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        def yarn_find_correction_range() -> tuple[int, int]:
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale: float = 1, ms: float = 1) -> float:
            if scale <= 1:
                return 1.0
            return 0.1 * ms * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val: int, max_val: int, dim: int) -> mx.array:
            if min_val == max_val:
                max_val += 0.001
            linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (max_val - min_val)
            return mx.clip(linear_func, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        if self.mscale != 1.0:
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return _apply_rope_safely(
            x,
            offset,
            lambda arr, off: mx.fast.rope(
                arr,
                self.dims,
                traditional=self.traditional,
                base=None,
                scale=1.0,
                offset=off,
                freqs=self._freqs,
            ),
        )


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: dict | None = None,
    max_position_embeddings: int | None = None,
) -> nn.Module:
    """Factory: create the appropriate RoPE module from config.

    Args:
        dims: Head dimension for rotation.
        base: Base frequency for RoPE.
        traditional: Use traditional (sincos) or interleaved layout.
        scaling_config: Optional dict with rope_type/type and scaling params.
        max_position_embeddings: Max sequence length (needed by some types).

    Returns:
        nn.Module implementing RoPE __call__(x, offset).
    """
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"

    if rope_type in ("default", "linear"):
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return DefaultRoPE(dims, traditional=traditional, base=base, scale=scale)

    if rope_type == "llama3":
        return Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings or 2048,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )

    if rope_type in ("yarn", "deepseek_yarn", "telechat3-yarn"):
        scaling_factor = scaling_config["factor"]
        rope_kwargs = {
            key: scaling_config[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
                "attn_factor",
            ]
            if key in scaling_config
        }
        return YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings or 2048,
            traditional=traditional,
            scaling_factor=scaling_factor,
            base=base,
            **rope_kwargs,
        )

    if rope_type == "longrope":
        return SuScaledRoPE(
            dims=dims,
            base=base,
            max_position_embeddings=max_position_embeddings or 131072,
            original_max_position_embeddings=scaling_config["original_max_position_embeddings"],
            short_factor=scaling_config["short_factor"],
            long_factor=scaling_config["long_factor"],
        )

    if rope_type == "mrope":
        return DefaultRoPE(dims, traditional=traditional, base=base)

    if rope_type == "dynamic":
        scale = scaling_config.get("factor", 2.0)
        scale = 1 / scale if isinstance(scale, (int, float)) and scale > 1 else 2.0
        return DynamicNTKScalingRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings or 32768,
            traditional=traditional,
            base=base,
            scale=scale,
        )

    raise ValueError(f"Unsupported RoPE type: {rope_type}")
