"""Minimal forward tests for olmo, olmo2, exaone, nemotron, plamo, helium (no weights)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlxs.load.registry import get_model_classes

# Minimal configs (small sizes) for forward-only tests. Aligned to config.json fields.
MINIMAL_CONFIGS = {
    "olmo": {
        "model_type": "olmo",
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "vocab_size": 256,
        "embedding_size": 256,
        "weight_tying": False,
    },
    "olmo2": {
        "model_type": "olmo2",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
    },
    "exaone": {
        "model_type": "exaone",
        "hidden_size": 64,
        "num_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "vocab_size": 256,
        "rope_theta": 10000.0,
        "layer_norm_epsilon": 1e-6,
        "num_key_value_heads": 4,
    },
    "nemotron": {
        "model_type": "nemotron",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "norm_eps": 1e-5,
        "vocab_size": 256,
    },
    "plamo": {
        "model_type": "plamo",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "n_shared_head": 2,
    },
    "helium": {
        "model_type": "helium",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "attention_bias": False,
        "head_dim": 16,
        "max_position_embeddings": 512,
        "mlp_bias": False,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    },
}


@pytest.mark.parametrize("model_type", ["olmo", "olmo2", "exaone", "nemotron", "plamo", "helium"])
def test_six_models_forward(model_type: str) -> None:
    """For each model: get_model_classes, from_dict(minimal_config), forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes(model_type)
    config = MINIMAL_CONFIGS[model_type]
    args = ArgsCls.from_dict(config)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers >= 1
    assert model.vocab_size == args.vocab_size
