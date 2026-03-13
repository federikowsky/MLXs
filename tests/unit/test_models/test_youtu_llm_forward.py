"""Minimal forward test for YoutuLLM (no weights, random init)."""

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_youtu_llm_forward_minimal() -> None:
    ModelCls, ArgsCls = get_model_classes("youtu_llm")
    minimal_config = {
        "model_type": "youtu_llm",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": 24,
        "kv_lora_rank": 16,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 8,
        "v_head_dim": 16,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_traditional": True,
        "vocab_size": 256,
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
