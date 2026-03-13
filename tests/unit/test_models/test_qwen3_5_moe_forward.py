"""qwen3_5_moe: registry, from_dict(minimal), make_cache(), one forward → logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_QWEN3_5_MOE = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "max_position_embeddings": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "shared_expert_intermediate_size": 64,
        "moe_intermediate_size": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "tie_word_embeddings": False,
    },
}


def test_qwen3_5_moe_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), one forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("qwen3_5_moe")
    args = ArgsCls.from_dict(MINIMAL_QWEN3_5_MOE)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, model.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
