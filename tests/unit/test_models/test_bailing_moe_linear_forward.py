"""Minimal forward test for bailing_moe_linear."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_BAILING_MOE_LINEAR = {
    "model_type": "bailing_moe_linear",
    "hidden_size": 64,
    "intermediate_size": 128,
    "max_position_embeddings": 128,
    "moe_intermediate_size": 32,
    "num_experts": 4,
    "num_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "vocab_size": 256,
    "first_k_dense_replace": 0,
    "layer_group_size": 2,
    "group_norm_size": 2,
}


def test_bailing_moe_linear_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("bailing_moe_linear")
    args = ArgsCls.from_dict(MINIMAL_BAILING_MOE_LINEAR)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 4
    assert model.vocab_size == 256
