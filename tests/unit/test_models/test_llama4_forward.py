"""Minimal forward test for llama4 (text-only config; mlx_lm llama4_text.py style)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

# Flat config (llama4_text): no text_config, no MoE, no chunking.
MINIMAL_LLAMA4 = {
    "model_type": "llama4",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "intermediate_size": 128,
    "intermediate_size_mlp": 128,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "rope_theta": 10000.0,
    "use_qk_norm": False,
    "no_rope_layers": [True, True],
}


def test_llama4_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("llama4")
    args = ArgsCls.from_dict(MINIMAL_LLAMA4)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256


def test_llama4_text_alias_forward() -> None:
    """Alias llama4_text resolves to llama4; same minimal config works."""
    ModelCls, ArgsCls = get_model_classes("llama4_text")
    args = ArgsCls.from_dict({**MINIMAL_LLAMA4, "model_type": "llama4_text"})
    model = ModelCls(args)
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3]]), cache=cache)
    assert logits.shape == (1, 3, 256)
