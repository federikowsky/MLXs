"""Minimal forward test for kimi_k25 (DeepSeek V3 text backbone)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_KIMI_K25 = {
    "model_type": "kimi_k25",
    "text_config": {
        "model_type": "deepseek_v3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 16,
        "v_head_dim": 32,
        "kv_lora_rank": 32,
        "q_lora_rank": 32,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "n_routed_experts": 4,
        "n_shared_experts": None,
        "num_experts_per_tok": 1,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 0,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "rope_theta": 10000.0,
    },
}


def test_kimi_k25_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("kimi_k25")
    args = ArgsCls.from_dict(MINIMAL_KIMI_K25)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    V = args.text_config.vocab_size
    assert logits.shape == (B, T, V)
    assert model.num_layers == 2
    assert model.vocab_size == V
