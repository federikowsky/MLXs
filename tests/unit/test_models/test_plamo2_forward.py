"""Minimal forward test for plamo2: registry, from_dict, make_cache, forward → logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_PLAMO2 = {
    "model_type": "plamo2",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "hidden_size_per_head": 16,
    "max_position_embeddings": 128,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_num_heads": 4,
    "mamba_step": 2,
    "mamba_enabled": True,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "tie_word_embeddings": False,
}


def test_plamo2_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), one forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("plamo2")
    args = ArgsCls.from_dict(MINIMAL_PLAMO2)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
