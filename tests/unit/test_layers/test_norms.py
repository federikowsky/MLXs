"""Unit tests for shared normalization layers — contract and smoke.

Covers GemmaRMSNorm (used by Gemma, Gemma2). Uses small shapes; requires MLX.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.layers.norms import GemmaRMSNorm


def test_gemma_rms_norm_shape() -> None:
    """GemmaRMSNorm(x) preserves shape."""
    norm = GemmaRMSNorm(dims=16, eps=1e-5)
    x = mx.ones((1, 4, 16), dtype=mx.float32)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    mx.eval(out)


def test_gemma_rms_norm_finite() -> None:
    """GemmaRMSNorm output is finite."""
    norm = GemmaRMSNorm(dims=8, eps=1e-6)
    x = mx.arange(16, dtype=mx.float32).reshape(1, 2, 8) * 0.1
    out = norm(x)
    mx.eval(out)
    assert mx.all(mx.isfinite(out)).item() is True


def test_gemma_rms_norm_eps() -> None:
    """GemmaRMSNorm accepts custom eps."""
    norm = GemmaRMSNorm(dims=4, eps=1e-4)
    x = mx.ones((1, 1, 4), dtype=mx.float32)
    out = norm(x)
    assert out.shape == x.shape
    mx.eval(out)


def test_gemma_rms_norm_weight_initialized() -> None:
    """GemmaRMSNorm has weight ones (1 + weight used in forward)."""
    norm = GemmaRMSNorm(dims=4)
    assert norm.weight.shape == (4,)
    mx.eval(norm.weight)
    assert mx.all(norm.weight == 1.0).item() is True
