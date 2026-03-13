"""Minimal forward test for gpt_oss (GPT-OSS MoE)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_GPT_OSS = {
    "model_type": "gpt_oss",
    "num_hidden_layers": 2,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 256,
    "rms_norm_eps": 1e-5,
    "hidden_size": 64,
    "intermediate_size": 128,
    "head_dim": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "sliding_window": 32,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "layer_types": ["sliding_attention", "full_attention"],
}


def test_gpt_oss_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("gpt_oss")
    args = ArgsCls.from_dict(MINIMAL_GPT_OSS)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
