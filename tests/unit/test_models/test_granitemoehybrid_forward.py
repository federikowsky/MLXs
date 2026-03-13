"""Minimal forward test for GraniteMoE Hybrid (no weights)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_GRANITEMOEHYBRID = {
    "model_type": "granitemoehybrid",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "max_position_embeddings": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "attention_bias": False,
    "embedding_multiplier": 1.0,
    "attention_multiplier": 1.0,
    "logits_scaling": 1.0,
    "residual_multiplier": 1.0,
    "layer_types": ["attention", "mamba"],
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "mamba_n_heads": 4,
    "mamba_d_head": 16,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_n_groups": 1,
    "mamba_proj_bias": False,
    "mamba_conv_bias": False,
}


def test_granitemoehybrid_forward() -> None:
    """Registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("granitemoehybrid")
    args = ArgsCls.from_dict(MINIMAL_GRANITEMOEHYBRID)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
