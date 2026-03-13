"""qwen3_vl_moe: registry, from_dict(minimal), make_cache(), forward -> logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_QWEN3_VL_MOE = {
    "model_type": "qwen3_vl_moe",
    "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "moe_intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "tie_word_embeddings": False,
        "max_position_embeddings": 256,
        "norm_topk_prob": True,
        "rope_theta": 10000.0,
    },
}


def test_qwen3_vl_moe_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("qwen3_vl_moe")
    args = ArgsCls.from_dict(MINIMAL_QWEN3_VL_MOE)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, model.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
