"""Minimal forward test for Jamba (no weights, random init)."""

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_jamba_forward_minimal() -> None:
    ModelCls, ArgsCls = get_model_classes("jamba")
    minimal_config = {
        "model_type": "jamba",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "attn_layer_offset": 0,
        "attn_layer_period": 2,
        "expert_layer_offset": 0,
        "expert_layer_period": 2,
        "mamba_d_conv": 4,
        "mamba_d_state": 8,
        "mamba_expand": 2,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 1024,
        "vocab_size": 256,
    }
    args = ArgsCls.from_dict(minimal_config)
    model = ModelCls(args)
    cache = model.make_cache()
    input_ids = mx.array([[1, 2, 3]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (1, 3, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
