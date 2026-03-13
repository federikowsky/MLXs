"""Minimal forward test for Gemma 3 (no weights, random init).

Port from mlx_lm gemma3_text; one forward → logits (B,T,V).
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_gemma3_forward_minimal() -> None:
    """get_model_classes, from_dict, Model, make_cache(), forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("gemma3")
    minimal_config = {
        "model_type": "gemma3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "num_key_value_heads": 2,
        "rope_theta": 10000.0,
        "rope_local_base_freq": 10000.0,
        "query_pre_attn_scalar": 16.0,
        "sliding_window": 64,
        "sliding_window_pattern": 2,
        "max_position_embeddings": 512,
        "rope_scaling": None,
    }
    args = ArgsCls.from_dict(minimal_config)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size


def test_gemma3_text_alias_forward() -> None:
    """Alias gemma3_text resolves to same model; forward produces (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("gemma3_text")
    config = {
        "model_type": "gemma3_text",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "num_key_value_heads": 2,
        "rope_theta": 10000.0,
        "rope_local_base_freq": 10000.0,
        "query_pre_attn_scalar": 16.0,
        "sliding_window": 64,
        "sliding_window_pattern": 2,
        "max_position_embeddings": 512,
        "rope_scaling": None,
    }
    args = ArgsCls.from_dict(config)
    model = ModelCls(args)
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3]]), cache=cache)
    assert logits.shape == (1, 3, args.vocab_size)
