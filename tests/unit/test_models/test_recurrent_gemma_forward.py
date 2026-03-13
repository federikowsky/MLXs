"""Minimal forward test for Recurrent Gemma (no weights, random init)."""

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_recurrent_gemma_forward_minimal() -> None:
    ModelCls, ArgsCls = get_model_classes("recurrent_gemma")
    minimal_config = {
        "model_type": "recurrent_gemma",
        "attention_bias": True,
        "conv1d_width": 4,
        "hidden_size": 64,
        "intermediate_size": 128,
        "logits_soft_cap": 0.0,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_window_size": 512,
        "vocab_size": 256,
        "block_types": ["recurrent", "attention"],
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
