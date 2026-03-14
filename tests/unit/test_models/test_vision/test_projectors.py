"""Tests for vision projectors (§7.4)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.models.vision.projectors import LinearProjector, MLPProjector


def test_mlp_projector_shape() -> None:
    """MLPProjector maps (N, D_in) -> (N, D_out)."""
    proj = MLPProjector(in_dim=64, hidden_dim=128, out_dim=32)
    mx.eval(proj.parameters())
    x = mx.zeros((10, 64))
    out = proj(x)
    assert out.shape == (10, 32)


def test_linear_projector_shape() -> None:
    """LinearProjector maps (N, D_in) -> (N, D_out)."""
    proj = LinearProjector(in_dim=64, out_dim=32)
    mx.eval(proj.parameters())
    x = mx.zeros((10, 64))
    out = proj(x)
    assert out.shape == (10, 32)


def test_mlp_projector_single_token() -> None:
    """MLPProjector handles single token input."""
    proj = MLPProjector(in_dim=16, hidden_dim=32, out_dim=8)
    mx.eval(proj.parameters())
    x = mx.zeros((1, 16))
    out = proj(x)
    assert out.shape == (1, 8)
