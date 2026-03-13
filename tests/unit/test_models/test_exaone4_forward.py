"""Minimal forward test for exaone4: registry, from_dict, make_cache, forward → logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_EXAONE4 = {
    "model_type": "exaone4",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def test_exaone4_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), one forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("exaone4")
    args = ArgsCls.from_dict(MINIMAL_EXAONE4)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
