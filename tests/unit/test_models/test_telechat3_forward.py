"""Minimal forward test for Telechat3 (no weights, random init).

Port from mlx_lm models/telechat3.py: GQA, RoPE (default or telechat3-yarn), SwiGLU, RMSNorm.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_TELECHAT3 = {
    "model_type": "telechat3",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "mlp_bias": False,
    "attention_bias": False,
    "tie_word_embeddings": False,
}


def test_telechat3_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("telechat3")
    args = ArgsCls.from_dict(MINIMAL_TELECHAT3)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
