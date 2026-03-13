"""Minimal forward test for mimo_v2_flash."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_MIMO_V2_FLASH = {
    "model_type": "mimo_v2_flash",
    "num_experts_per_tok": 1,
    "hybrid_layer_pattern": [0, 1],
    "moe_layer_freq": [0, 1],
    "add_swa_attention_sink_bias": False,
    "add_full_attention_sink_bias": False,
    "sliding_window_size": 4,
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": None,
    "n_routed_experts": 2,
    "routed_scaling_factor": 1.0,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": False,
    "n_group": 1,
    "topk_group": 1,
    "max_position_embeddings": 128,
    "layernorm_epsilon": 1e-6,
    "rope_theta": 10000.0,
    "swa_rope_theta": 10000.0,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 4,
    "head_dim": 16,
    "v_head_dim": 16,
    "swa_head_dim": 16,
    "swa_v_head_dim": 16,
    "partial_rotary_factor": 1,
}


def test_mimo_v2_flash_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("mimo_v2_flash")
    args = ArgsCls.from_dict(MINIMAL_MIMO_V2_FLASH)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
