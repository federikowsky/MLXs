"""Minimal forward test for SmolLM3 (no weights, random init)."""

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_smollm3_forward_minimal() -> None:
    ModelCls, ArgsCls = get_model_classes("smollm3")
    minimal_config = {
        "model_type": "smollm3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5,
        "vocab_size": 256,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "no_rope_layer_interval": 4,
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
