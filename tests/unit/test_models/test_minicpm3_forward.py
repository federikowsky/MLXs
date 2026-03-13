"""Minimal forward test for MiniCPM3 (no weights, random init)."""

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_minicpm3_forward_minimal() -> None:
    ModelCls, ArgsCls = get_model_classes("minicpm3")
    minimal_config = {
        "model_type": "minicpm3",
        "hidden_size": 64,
        "dim_model_base": 1,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "kv_lora_rank": 8,
        "rms_norm_eps": 1e-5,
        "scale_depth": 1.0,
        "scale_emb": 1.0,
        "max_position_embeddings": 2048,
        "vocab_size": 256,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 4096,
            "short_factor": 1.0,
            "long_factor": 1.0,
        },
        "tie_word_embeddings": False,
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
