"""Minimal forward test for afm7."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_AFM7 = {
    "model_type": "afm7",
    "vocab_size": 256,
    "hidden_dim": 64,
    "num_layers": 4,
    "num_kv_reuse_layers": 1,
    "num_heads": 4,
    "num_kv_heads": 4,
    "hidden_dim_scale_factor": 3.25,
    "rope_theta": 50000.0,
    "rms_norm_eps": 1e-5,
}


def test_afm7_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("afm7")
    args = ArgsCls.from_dict(MINIMAL_AFM7)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 4
    assert model.vocab_size == 256
