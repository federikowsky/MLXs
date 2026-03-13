"""Minimal forward test for glm4_moe (no weights)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_GLM4_MOE = {
    "model_type": "glm4_moe",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "max_position_embeddings": 512,
    "moe_intermediate_size": 96,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "n_group": 1,
    "head_dim": 16,
    "topk_group": 1,
    "n_shared_experts": None,
    "n_routed_experts": 4,
    "routed_scaling_factor": 1.0,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 0,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "use_qk_norm": False,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "partial_rotary_factor": 1.0,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
}


def test_glm4_moe_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("glm4_moe")
    args = ArgsCls.from_dict(MINIMAL_GLM4_MOE)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
