"""Minimal forward and support tests for Ouro (registry, from_dict, make_cache, sanitize)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

# Minimal config: 2 layers, 2 ut_steps (cache length 4).
MINIMAL_OURO = {
    "model_type": "ouro",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "total_ut_steps": 2,
}


def test_ouro_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("ouro")
    args = ArgsCls.from_dict(MINIMAL_OURO)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256


def test_ouro_make_cache_length() -> None:
    """make_cache returns num_hidden_layers * total_ut_steps KVCache slots."""
    ModelCls, ArgsCls = get_model_classes("ouro")
    args = ArgsCls.from_dict(MINIMAL_OURO)
    model = ModelCls(args)
    cache = model.make_cache()
    assert len(cache) == 2 * 2  # 2 layers, 2 ut_steps


def test_ouro_sanitize_dequantize() -> None:
    """sanitize removes rotary keys and dequantizes 4-bit weight+scales+biases."""
    ModelCls, ArgsCls = get_model_classes("ouro")
    args = ArgsCls.from_dict(MINIMAL_OURO)
    model = ModelCls(args)

    # Use mx.quantize output so shapes match what mx.dequantize expects
    w_float = mx.ones((64, 64), dtype=mx.float32)
    w_packed, scales, biases = mx.quantize(w_float, group_size=64, bits=4)

    weights = {
        "model.layers.0.self_attn.q_proj.weight": w_packed,
        "model.layers.0.self_attn.q_proj.scales": scales,
        "model.layers.0.self_attn.q_proj.biases": biases,
        "model.embed_tokens.weight": mx.zeros((256, 64)),
        "rotary_emb.inv_freq": mx.zeros((1,)),
    }
    out = model.sanitize(weights)

    assert "rotary_emb.inv_freq" not in out
    assert "model.layers.0.self_attn.q_proj.scales" not in out
    assert "model.layers.0.self_attn.q_proj.biases" not in out
    assert "model.layers.0.self_attn.q_proj.weight" in out
    assert out["model.layers.0.self_attn.q_proj.weight"].dtype == mx.float32
    assert "model.embed_tokens.weight" in out
