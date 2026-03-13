"""Multi-Latent Attention (MLA) helper layers.

Provides MultiLinear and QuantizedMultiLinear for DeepSeek V3's compressed
KV attention. Compatible with mlx_lm mla.py API.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class MultiLinear(nn.Module):
    """Batched per-head linear projection."""

    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_heads, output_dims, input_dims),
        )

    def __call__(self, x: mx.array, transpose: bool = True) -> mx.array:
        if transpose:
            return x @ self.weight.swapaxes(-1, -2)
        return x @ self.weight

    def to_quantized(
        self,
        group_size: int,
        bits: int,
        mode: str = "affine",
    ) -> QuantizedMultiLinear:
        num_heads, output_dims, input_dims = self.weight.shape
        ql = QuantizedMultiLinear(input_dims, output_dims, num_heads, group_size, bits, mode)
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight,
            group_size,
            bits,
            mode=mode,
        )
        ql.biases = biases[0] if biases else None
        return ql


class QuantizedMultiLinear(nn.Module):
    """Quantized batched per-head linear projection."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_heads: int,
        group_size: int,
        bits: int,
        mode: str,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        scale = math.sqrt(1.0 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_heads, output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(weight, group_size, bits, mode=mode)
        self.biases = biases[0] if biases else None
        self.freeze()

    def __call__(self, x: mx.array, transpose: bool = True) -> mx.array:
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=transpose,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
