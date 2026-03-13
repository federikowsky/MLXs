"""Minimal forward test for phixtral (Phi-style MoE)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_PHIXTRAL = {
    "model_type": "phixtral",
    "num_vocab": 256,
    "model_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "rotary_dim": 16,
    "num_experts_per_tok": 2,
    "num_local_experts": 4,
}


def test_phixtral_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("phixtral")
    args = ArgsCls.from_dict(MINIMAL_PHIXTRAL)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.num_vocab)
    assert model.num_layers == 2
    assert model.vocab_size == 256
