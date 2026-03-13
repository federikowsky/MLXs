"""Minimal forward test for baichuan_m1 (global + SWA, conv on K/V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_BAICHUAN_M1 = {
    "model_type": "baichuan_m1",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "rope_theta": 10000.0,
    "sliding_window": 128,
    "sliding_window_layers": [1],
    "conv_window": 2,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def test_baichuan_m1_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("baichuan_m1")
    args = ArgsCls.from_dict(MINIMAL_BAICHUAN_M1)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
