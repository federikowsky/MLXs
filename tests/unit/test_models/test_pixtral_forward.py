"""Minimal forward test for Pixtral (text backbone only; no weights).

get_model_classes("pixtral"), ModelArgs.from_dict(minimal), Model(args),
make_cache(), forward → logits (B,T,V).
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_pixtral_forward_minimal() -> None:
    """get_model_classes, from_dict, Model, make_cache(), forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("pixtral")
    minimal_config = {
        "model_type": "pixtral",
        "text_config": {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        },
    }
    args = ArgsCls.from_dict(minimal_config)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.text_config["vocab_size"])
    assert model.num_layers == 2
    assert model.vocab_size == 256
