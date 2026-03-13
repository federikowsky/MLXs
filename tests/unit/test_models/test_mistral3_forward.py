"""Minimal forward test for mistral3 (no weights)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_MISTRAL3 = {
    "model_type": "mistral3",
    "text_config": {
        "model_type": "ministral3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "layer_types": ["full_attention", "full_attention"],
    },
}


def test_mistral3_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("mistral3")
    args = ArgsCls.from_dict(MINIMAL_MISTRAL3)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, model.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
