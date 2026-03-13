"""Minimal forward test for lille_130m (RoPE, GQA, SwiGLU)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_LILLE_130M = {
    "model_type": "lille_130m",
    "block_size": 512,
    "layer_norm_eps": 1e-5,
    "n_embd": 256,
    "n_head": 4,
    "n_kv_heads": 4,
    "n_layer": 2,
    "rope_theta": 10000.0,
    "vocab_size": 1024,
}


def test_lille_130m_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("lille_130m")
    args = ArgsCls.from_dict(MINIMAL_LILLE_130M)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 1024
