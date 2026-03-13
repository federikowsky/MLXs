"""Minimal forward test for minimax (MoE + QK-norm)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_MINIMAX = {
    "model_type": "minimax",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 128,
    "num_experts_per_tok": 1,
    "num_local_experts": 4,
    "shared_intermediate_size": 0,
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rotary_dim": 16,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "scoring_func": "sigmoid",
    "head_dim": None,
    "use_qk_norm": True,
}


def test_minimax_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("minimax")
    args = ArgsCls.from_dict(MINIMAL_MINIMAX)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
