"""Minimal forward test for glm4_moe_lite (no weights)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_GLM4_MOE_LITE = {
    "model_type": "glm4_moe_lite",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "routed_scaling_factor": 1.8,
    "kv_lora_rank": 32,
    "q_lora_rank": 48,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 24,
    "v_head_dim": 32,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 1,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "partial_rotary_factor": 1.0,
    "tie_word_embeddings": False,
    "num_nextn_predict_layers": 1,
}


def test_glm4_moe_lite_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("glm4_moe_lite")
    args = ArgsCls.from_dict(MINIMAL_GLM4_MOE_LITE)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
