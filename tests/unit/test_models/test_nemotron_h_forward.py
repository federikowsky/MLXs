"""Minimal forward test for nemotron_h: registry, from_dict, make_cache, forward → logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_NEMOTRON_H = {
    "model_type": "nemotron_h",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "max_position_embeddings": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "attention_bias": False,
    "mamba_num_heads": 4,
    "mamba_head_dim": 16,
    "mamba_proj_bias": False,
    "ssm_state_size": 8,
    "conv_kernel": 4,
    "n_groups": 1,
    "mlp_bias": False,
    "layer_norm_epsilon": 1e-5,
    "use_bias": False,
    "use_conv_bias": False,
    "hybrid_override_pattern": ["M", "*"],
}


def test_nemotron_h_forward() -> None:
    """get_model_classes('nemotron_h'), from_dict(minimal), Model, make_cache(), forward → (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("nemotron_h")
    args = ArgsCls.from_dict(MINIMAL_NEMOTRON_H)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
