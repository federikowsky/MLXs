"""Minimal forward test for InternLM3 (no weights, random init).

Port from mlx_lm models/internlm3.py: separate q/k/v/o, RMSNorm, SwiGLU, Dynamic NTK RoPE.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_INTERNLM3 = {
    "model_type": "internlm3",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "bias": False,
    "qkv_bias": False,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def test_internlm3_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("internlm3")
    args = ArgsCls.from_dict(MINIMAL_INTERNLM3)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
