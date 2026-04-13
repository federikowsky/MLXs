"""Unit tests for shared RoPE layers — contract and smoke.

Covers initialize_rope factory and RoPE classes: default/linear, Llama3RoPE,
YarnRoPE, SuScaledRoPE, DynamicNTKScalingRoPE. Uses small shapes for speed.
"""

from __future__ import annotations

import mlx.core as mx

import mlxs.layers.rope as rope_mod
from mlxs.layers.rope import (
    DefaultRoPE,
    DynamicNTKScalingRoPE,
    Llama3RoPE,
    SuScaledRoPE,
    YarnRoPE,
    initialize_rope,
)


def _rope_input(shape: tuple[int, ...] = (1, 2, 8)) -> mx.array:
    """Small float32 input for RoPE (batch, seq, dims)."""
    return mx.ones(shape, dtype=mx.float32)


def test_initialize_rope_default() -> None:
    """initialize_rope with scaling_config=None returns default wrapper."""
    rope = initialize_rope(dims=8, base=10000.0, traditional=False, scaling_config=None)
    assert isinstance(rope, DefaultRoPE)
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    mx.eval(out)


def test_initialize_rope_linear() -> None:
    """initialize_rope with type linear uses factor for scale."""
    rope = initialize_rope(
        dims=8,
        base=10000.0,
        traditional=False,
        scaling_config={"type": "linear", "factor": 2.0},
    )
    assert isinstance(rope, DefaultRoPE)
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    mx.eval(out)


def test_initialize_rope_llama3() -> None:
    """initialize_rope with type llama3 returns Llama3RoPE."""
    rope = initialize_rope(
        dims=8,
        base=10000.0,
        traditional=False,
        scaling_config={
            "type": "llama3",
            "factor": 2.0,
            "original_max_position_embeddings": 8192,
        },
        max_position_embeddings=4096,
    )
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    mx.eval(out)


def test_initialize_rope_yarn() -> None:
    """initialize_rope with type yarn returns YarnRoPE."""
    rope = initialize_rope(
        dims=8,
        base=10000.0,
        traditional=False,
        scaling_config={
            "type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 4096,
        },
        max_position_embeddings=8192,
    )
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    mx.eval(out)


def test_initialize_rope_longrope() -> None:
    """initialize_rope with type longrope returns SuScaledRoPE."""
    rope = initialize_rope(
        dims=8,
        base=10000.0,
        traditional=False,
        scaling_config={
            "type": "longrope",
            "original_max_position_embeddings": 4096,
            "short_factor": 1.0,
            "long_factor": 4.0,
        },
        max_position_embeddings=131072,
    )
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    mx.eval(out)


def test_initialize_rope_dynamic() -> None:
    """initialize_rope with type dynamic returns DynamicNTKScalingRoPE."""
    rope = initialize_rope(
        dims=8,
        base=10000.0,
        traditional=False,
        scaling_config={"type": "dynamic", "factor": 2.0},
        max_position_embeddings=32768,
    )
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    mx.eval(out)


def test_initialize_rope_unsupported_raises() -> None:
    """initialize_rope with unknown type raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unsupported RoPE type"):
        initialize_rope(
            dims=8,
            base=10000.0,
            traditional=False,
            scaling_config={"type": "unknown_type"},
        )


def test_llama3_rope_call_with_offset() -> None:
    """Llama3RoPE __call__ preserves shape with offset."""
    rope = Llama3RoPE(
        dims=8,
        max_position_embeddings=2048,
        traditional=False,
        base=10000.0,
        scaling_config={
            "factor": 1.0,
            "original_max_position_embeddings": 2048,
        },
    )
    x = _rope_input((1, 2, 8))
    out = rope(x, offset=10)
    assert out.shape == x.shape
    mx.eval(out)


def test_yarn_rope_call() -> None:
    """YarnRoPE __call__ preserves shape."""
    rope = YarnRoPE(
        dims=8,
        max_position_embeddings=2048,
        traditional=False,
        base=10000.0,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
    )
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    mx.eval(out)


def test_su_scaled_rope_call() -> None:
    """SuScaledRoPE __call__ preserves shape."""
    rope = SuScaledRoPE(
        dims=8,
        base=10000.0,
        max_position_embeddings=8192,
        original_max_position_embeddings=4096,
        short_factor=1.0,
        long_factor=2.0,
    )
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    mx.eval(out)


def test_dynamic_ntk_rope_call() -> None:
    """DynamicNTKScalingRoPE __call__ preserves shape."""
    rope = DynamicNTKScalingRoPE(
        dims=8,
        max_position_embeddings=2048,
        traditional=False,
        base=10000.0,
        scale=1.0,
    )
    x = _rope_input((1, 2, 8))
    out = rope(x, 0)
    assert out.shape == x.shape
    mx.eval(out)


def test_apply_rope_safely_uses_rowwise_path_for_batched_decode() -> None:
    """Batched single-token decode with offset>0 is routed row-wise."""
    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_apply(x: mx.array, offset: int | mx.array) -> mx.array:
        offset_int = int(offset) if isinstance(offset, int) else int(offset.item())
        calls.append((tuple(int(dim) for dim in x.shape), offset_int))
        out = x + 1
        if x.shape[0] > 1:
            out = mx.concatenate([out[:1], out[1:2] + 5], axis=0)
        return out

    x = _rope_input((2, 4, 1, 8))
    out = rope_mod._apply_rope_safely(x, 9, fake_apply)
    assert calls == [((1, 4, 1, 8), 9), ((1, 4, 1, 8), 9)]
    assert bool(mx.allclose(out[0], out[1], atol=0, rtol=0).item())


def test_apply_rope_safely_keeps_prefill_batched_path() -> None:
    """Prefill and zero-offset paths still use the batched fast path."""
    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_apply(x: mx.array, offset: int | mx.array) -> mx.array:
        offset_int = int(offset) if isinstance(offset, int) else int(offset.item())
        calls.append((tuple(int(dim) for dim in x.shape), offset_int))
        return x

    prefill = _rope_input((2, 4, 2, 8))
    out_prefill = rope_mod._apply_rope_safely(prefill, 9, fake_apply)
    assert out_prefill.shape == prefill.shape
    zero_offset = _rope_input((2, 4, 1, 8))
    out_zero = rope_mod._apply_rope_safely(zero_offset, 0, fake_apply)
    assert out_zero.shape == zero_offset.shape
    assert calls == [((2, 4, 2, 8), 9), ((2, 4, 1, 8), 0)]
