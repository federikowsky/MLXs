"""Multi-modal projectors — map vision encoder output to LM hidden dim.

Used by VL model wrappers to project vision features into the language
model embedding space before merge (§7.4).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MLPProjector(nn.Module):
    """Two-layer MLP projector with GELU activation.

    Used by Qwen2-VL, Qwen3-VL, KimiVL.

    Architecture: Linear(in_dim, hidden_dim) -> GELU -> Linear(hidden_dim, out_dim)
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class LinearProjector(nn.Module):
    """Single linear projection.

    Used by Pixtral and other models with simple vision-to-LM mapping.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)
