"""Minimal forward test for dots1 (MoE + top-k router, optional shared experts)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_DOTS1 = {
    "model_type": "dots1",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "num_key_value_heads": 4,
    "first_k_dense_replace": 0,
    "moe_intermediate_size": 64,
    "n_routed_experts": 4,
    "n_shared_experts": 0,
    "norm_topk_prob": True,
    "num_experts_per_tok": 1,
    "rope_theta": 10000.0,
    "routed_scaling_factor": 1.0,
    "n_group": 1,
    "topk_group": 1,
}


def test_dots1_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("dots1")
    args = ArgsCls.from_dict(MINIMAL_DOTS1)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
