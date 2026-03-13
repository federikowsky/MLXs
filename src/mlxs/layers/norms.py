"""Shared normalization layers used by multiple model architectures.

Only norms used by more than one model belong here. Model-specific norms
(e.g. LayerNorm1P for Nemotron, LayerNormNoAffine for OLMo) live in the
respective model module to keep SRP: one reason to change per norm.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: scale is (1 + weight) instead of weight.

    Used by Gemma and Gemma 2. Same as nn.RMSNorm but with 1 + weight
    so the learned weight is a residual adjustment.
    """

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
