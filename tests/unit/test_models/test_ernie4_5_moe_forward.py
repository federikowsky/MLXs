"""Minimal forward test for ernie4_5_moe."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_ERNIE4_5_MOE = {
    "model_type": "ernie4_5_moe",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "use_bias": False,
    "tie_word_embeddings": False,
    "moe_num_experts": 4,
    "moe_layer_start_index": 0,
    "moe_layer_end_index": 1,
    "moe_k": 1,
    "moe_layer_interval": 1,
    "moe_num_shared_experts": 0,
    "moe_gate_act": "softmax",
}


def test_ernie4_5_moe_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("ernie4_5_moe")
    args = ArgsCls.from_dict(MINIMAL_ERNIE4_5_MOE)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
