"""Unit tests for shared activation functions — contract and smoke.

Covers swiglu, relu_squared, xielu (functional and XieLU module).
Uses small shapes; requires MLX.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.layers.activations import XieLU, relu_squared, swiglu, xielu


def test_swiglu_shape_and_finite() -> None:
    """swiglu(gate, x) returns same shape, finite values."""
    gate = mx.ones((1, 4, 16), dtype=mx.float32)
    x = mx.ones((1, 4, 16), dtype=mx.float32)
    out = swiglu(gate, x)
    assert out.shape == gate.shape
    assert out.dtype == gate.dtype
    mx.eval(out)
    assert mx.all(mx.isfinite(out)).item() is True


def test_swiglu_different_inputs() -> None:
    """swiglu with non-constant inputs produces different output."""
    gate = mx.arange(8, dtype=mx.float32).reshape(1, 2, 4)
    x = mx.arange(8, dtype=mx.float32).reshape(1, 2, 4) * 0.1
    out = swiglu(gate, x)
    assert out.shape == gate.shape
    mx.eval(out)
    assert mx.all(mx.isfinite(out)).item() is True


def test_relu_squared_shape_and_finite() -> None:
    """relu_squared(x) returns same shape, finite values."""
    x = mx.array([[-1.0, 0.0, 1.0, 2.0]], dtype=mx.float32)
    out = relu_squared(x)
    assert out.shape == x.shape
    mx.eval(out)
    assert mx.all(mx.isfinite(out)).item() is True


def test_relu_squared_zeros_for_negative() -> None:
    """relu_squared is zero for negative input."""
    x = mx.array([[-2.0, -1.0]], dtype=mx.float32)
    out = relu_squared(x)
    mx.eval(out)
    assert mx.all(out == 0).item() is True


def test_xielu_shape_and_finite() -> None:
    """xielu(x, alpha_p, alpha_n, beta, eps) returns same shape, finite."""
    x = mx.ones((1, 2, 8), dtype=mx.float32)
    alpha_p = mx.array(0.5)
    alpha_n = mx.array(0.3)
    beta = mx.array(0.1)
    eps = mx.array(-1e-6)
    out = xielu(x, alpha_p, alpha_n, beta, eps)
    assert out.shape == x.shape
    mx.eval(out)
    assert mx.all(mx.isfinite(out)).item() is True


def test_xielu_negative_branch() -> None:
    """xielu with negative x uses negative branch (finite)."""
    x = mx.array([[-1.0, -2.0]], dtype=mx.float32)
    alpha_p = mx.array(0.8)
    alpha_n = mx.array(0.8)
    beta = mx.array(0.5)
    eps = mx.array(-1e-6)
    out = xielu(x, alpha_p, alpha_n, beta, eps)
    assert out.shape == x.shape
    mx.eval(out)
    assert mx.all(mx.isfinite(out)).item() is True


def test_xielu_module_shape_and_finite() -> None:
    """XieLU module __call__(x) returns same shape, finite."""
    act = XieLU(alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6)
    x = mx.ones((1, 4, 16), dtype=mx.float32)
    out = act(x)
    assert out.shape == x.shape
    mx.eval(out)
    assert mx.all(mx.isfinite(out)).item() is True


def test_xielu_module_default_init() -> None:
    """XieLU() with defaults runs without error."""
    act = XieLU()
    x = mx.zeros((1, 2, 8), dtype=mx.float32)
    out = act(x)
    assert out.shape == x.shape
    mx.eval(out)
