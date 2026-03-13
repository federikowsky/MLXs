"""Minimal forward test for qwen3_5: registry, from_dict, make_cache, forward -> logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_QWEN3_5 = {
    "model_type": "qwen3_5",
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
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "partial_rotary_factor": 0.25,
}


def test_qwen3_5_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("qwen3_5")
    args = ArgsCls.from_dict(MINIMAL_QWEN3_5)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
