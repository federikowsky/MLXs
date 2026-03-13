"""Minimal forward test for Gemma 3N (no weights, random init).

Port from mlx_lm gemma3n; one forward → logits (B,T,V).
"""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_gemma3n_forward_minimal() -> None:
    """get_model_classes, from_dict, Model, make_cache(), forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("gemma3n")
    minimal_config = {
        "model_type": "gemma3n",
        "text_config": {
            "model_type": "gemma3n",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 0,
            "vocab_size_per_layer_input": 256,
            "sliding_window": 64,
            "max_position_embeddings": 512,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 10000.0,
            "final_logit_softcapping": None,
            "layer_types": ["full_attention", "full_attention"],
            "activation_sparsity_pattern": None,
            "hidden_size_per_layer_input": 32,
            "altup_num_inputs": 2,
            "altup_coef_clip": None,
            "altup_correct_scale": False,
            "altup_active_idx": 0,
            "laurel_rank": 16,
            "rope_scaling": None,
        },
    }
    args = ArgsCls.from_dict(minimal_config)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    vocab_size = model.vocab_size
    assert logits.shape == (B, T, vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
