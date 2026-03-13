"""Shared activation functions for model architectures.

Provides compiled SwiGLU and XieLU activations compatible with mlx_lm.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    """SwiGLU activation: silu(gate) * x."""
    return nn.silu(gate) * x


@partial(mx.compile, shapeless=True)
def relu_squared(x: mx.array) -> mx.array:
    """ReLU squared: relu(x)^2. Used by Nemotron MLP."""
    return nn.relu(x).square()


@partial(mx.compile, shapeless=True)
def xielu(
    x: mx.array,
    alpha_p: mx.array,
    alpha_n: mx.array,
    beta: mx.array,
    eps: mx.array,
) -> mx.array:
    """XieLU activation with learnable parameters."""
    alpha_p = nn.softplus(alpha_p)
    alpha_n = beta + nn.softplus(alpha_n)
    return mx.where(
        x > 0,
        alpha_p * mx.square(x) + beta * x,
        (mx.expm1(mx.minimum(x, eps)) - x) * alpha_n + beta * x,
    )


class XieLU(nn.Module):
    """XieLU activation with learnable parameters (nn.Module)."""

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
    ) -> None:
        super().__init__()
        alpha_p_tensor = mx.array(alpha_p_init)
        alpha_n_tensor = mx.array(alpha_n_init - beta)
        self.alpha_p = mx.log(mx.exp(alpha_p_tensor) - 1)
        self.alpha_n = mx.log(mx.exp(alpha_n_tensor) - 1)
        self.beta = mx.array(beta)
        self.eps = mx.array(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return xielu(x, self.alpha_p, self.alpha_n, self.beta, self.eps)
