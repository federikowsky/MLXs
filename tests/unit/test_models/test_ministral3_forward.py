"""Minimal forward test for Ministral3 (no weights, random init).

Structure follows mlx_lm models/ministral3.py: dense decoder with optional
sliding-window layers, Llama 4 attention scaling, RoPE, SwiGLU.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_ministral3_forward_minimal() -> None:
    """get_model_classes, from_dict(minimal), Model, make_cache, one forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("ministral3")
    minimal_config = {
        "model_type": "ministral3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 4096,
        },
        "tie_word_embeddings": False,
        "layer_types": ["full_attention", "full_attention"],
    }
    args = ArgsCls.from_dict(minimal_config)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
