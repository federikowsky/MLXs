"""Minimal forward test for phi3small (no weights)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_PHI3SMALL = {
    "model_type": "phi3small",
    "hidden_size": 64,
    "dense_attention_every_n_layers": 2,
    "ff_intermediate_size": 128,
    "gegelu_limit": 256.0,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "layer_norm_epsilon": 1e-5,
    "vocab_size": 256,
    "num_key_value_heads": 4,
    "mup_attn_multiplier": 1.0,
    "mup_use_scaling": True,
    "mup_embedding_multiplier": 10.0,
    "mup_width_multiplier": 8.0,
    "rope_embedding_base": 1000000.0,
    "rope_position_scale": 1.0,
    "blocksparse_block_size": 64,
    "blocksparse_num_local_blocks": 16,
    "blocksparse_vert_stride": 8,
}


def test_phi3small_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("phi3small")
    args = ArgsCls.from_dict(MINIMAL_PHI3SMALL)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
