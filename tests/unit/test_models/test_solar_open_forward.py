"""Minimal forward test for solar_open (GLM-4 MoE-compatible config)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_SOLAR_OPEN = {
    "model_type": "solar_open",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "n_shared_experts": 0,
    "n_routed_experts": 4,
    "routed_scaling_factor": 1.0,
    "num_experts_per_tok": 1,
    "first_k_dense_replace": 0,
    "norm_topk_prob": True,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "partial_rotary_factor": 1.0,
    "rope_scaling": None,
    "attention_bias": False,
    "use_qk_norm": False,
    "n_group": 1,
    "topk_group": 1,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
}


def test_solar_open_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("solar_open")
    args = ArgsCls.from_dict(MINIMAL_SOLAR_OPEN)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
