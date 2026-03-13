"""Minimal forward test for step3p5."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_STEP3P5 = {
    "model_type": "step3p5",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "vocab_size": 256,
    "num_attention_heads": 4,
    "num_attention_groups": 4,
    "head_dim": 16,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "sliding_window": 128,
    "use_head_wise_attn_gate": True,
    "moe_num_experts": 4,
    "moe_top_k": 2,
    "moe_intermediate_size": 64,
    "share_expert_dim": 64,
    "moe_router_scaling_factor": 1.0,
    "norm_expert_weight": True,
    "moe_layers_enum": "1",
}


def test_step3p5_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("step3p5")
    args = ArgsCls.from_dict(MINIMAL_STEP3P5)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
