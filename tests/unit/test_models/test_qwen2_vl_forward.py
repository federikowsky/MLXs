"""Minimal forward test for qwen2_vl (Qwen2 text backbone)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

# Flat minimal config: from_dict uses params as text_config (mlx_lm compatibility)
MINIMAL_QWEN2_VL = {
    "model_type": "qwen2_vl",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
}


def test_qwen2_vl_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("qwen2_vl")
    args = ArgsCls.from_dict(MINIMAL_QWEN2_VL)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    V = args.text_config["vocab_size"]
    assert logits.shape == (B, T, V)
    assert model.num_layers == 2
    assert model.vocab_size == V
