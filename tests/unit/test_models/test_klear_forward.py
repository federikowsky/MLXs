"""Minimal forward test for klear (MoE with optional MLP-only layers, shared experts)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_KLEAR = {
    "model_type": "klear",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "attention_bias": False,
    "mlp_only_layers": [],
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "decoder_sparse_step": 2,
    "n_shared_experts": 1,
    "moe_intermediate_size": 64,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "num_key_value_heads": 4,
    "rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "norm_topk_prob": True,
}


def test_klear_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("klear")
    args = ArgsCls.from_dict(MINIMAL_KLEAR)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
