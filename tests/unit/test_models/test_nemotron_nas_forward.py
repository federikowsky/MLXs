"""Minimal forward test for nemotron_nas: registry, from_dict, make_cache, forward → logits."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_NEMOTRON_NAS = {
    "model_type": "nemotron_nas",
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
    "block_configs": [
        {
            "attention": {"no_op": False, "replace_with_linear": False, "n_heads_in_group": 2},
            "ffn": {"no_op": False, "replace_with_linear": False, "ffn_mult": 1.5},
        },
        {
            "attention": {"no_op": False, "replace_with_linear": False, "n_heads_in_group": 2},
            "ffn": {"no_op": False, "replace_with_linear": False, "ffn_mult": 1.5},
        },
    ],
}


def test_nemotron_nas_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), forward → (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("nemotron_nas")
    args = ArgsCls.from_dict(MINIMAL_NEMOTRON_NAS)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
