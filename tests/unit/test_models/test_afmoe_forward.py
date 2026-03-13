"""Minimal forward test for afmoe."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_AFMOE = {
    "model_type": "afmoe",
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "num_shared_experts": 0,
    "num_dense_layers": 1,
    "route_norm": True,
    "route_scale": 2.0,
    "score_func": "sigmoid",
    "n_group": 1,
    "topk_group": 1,
    "sliding_window": 64,
    "mup_enabled": False,
    "layer_types": ["full_attention", "full_attention"],
}


def test_afmoe_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("afmoe")
    args = ArgsCls.from_dict(MINIMAL_AFMOE)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 512
