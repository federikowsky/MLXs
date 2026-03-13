"""exaone_moe: registry, from_dict, make_cache, one forward → logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_EXAONE_MOE = {
    "model_type": "exaone_moe",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "num_shared_experts": 0,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "sliding_window": 128,
    "layer_types": ["global", "global"],
    "is_moe_layer": [True, False],
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def test_exaone_moe_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), one forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("exaone_moe")
    args = ArgsCls.from_dict(MINIMAL_EXAONE_MOE)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
