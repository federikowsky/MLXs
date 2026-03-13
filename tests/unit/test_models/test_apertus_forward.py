"""Minimal forward test for apertus."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_APERTUS = {
    "model_type": "apertus",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "mlp_bias": False,
    "num_attention_heads": 4,
    "attention_bias": False,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "num_key_value_heads": 4,
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
    "post_norm": True,
    "qk_norm": True,
    "tie_word_embeddings": False,
    "rope_traditional": False,
    "rope_scaling": None,
}


def test_apertus_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("apertus")
    args = ArgsCls.from_dict(MINIMAL_APERTUS)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
