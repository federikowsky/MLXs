"""Minimal forward test for nanochat (RoPE, QK-norm, ReLU², softcap)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_NANOCHAT = {
    "model_type": "nanochat",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "intermediate_size": 128,
    "rope_theta": 10000.0,
}


def test_nanochat_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("nanochat")
    args = ArgsCls.from_dict(MINIMAL_NANOCHAT)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
