"""Minimal forward test for lfm2_moe (short conv + full attention + dense/MoE FFN)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_LFM2_MOE = {
    "model_type": "lfm2_moe",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 128,
    "use_expert_bias": False,
    "num_dense_layers": 2,
    "norm_eps": 1e-6,
    "conv_bias": False,
    "conv_L_cache": 4,
    "rope_theta": 10000.0,
    "layer_types": ["full_attention", "short_conv", "full_attention", "short_conv"],
}


def test_lfm2_moe_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("lfm2_moe")
    args = ArgsCls.from_dict(MINIMAL_LFM2_MOE)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 4
    assert model.vocab_size == 256
