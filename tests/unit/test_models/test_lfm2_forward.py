"""Minimal forward test for lfm2 (short conv + full attention)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_LFM2 = {
    "model_type": "lfm2",
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 128,
    "norm_eps": 1e-6,
    "conv_bias": False,
    "conv_L_cache": 4,
    "block_dim": 64,
    "block_ff_dim": 128,
    "block_multiple_of": 64,
    "block_ffn_dim_multiplier": None,
    "block_auto_adjust_ff_dim": True,
    "rope_theta": 10000.0,
    "layer_types": ["full_attention", "short_conv"],
}


def test_lfm2_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("lfm2")
    args = ArgsCls.from_dict(MINIMAL_LFM2)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
