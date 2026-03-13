"""Minimal forward test for hunyuan (MoE + CLA + QK-norm + NTK RoPE)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_HUNYUAN = {
    "model_type": "hunyuan",
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "attention_bias": False,
    "moe_topk": 2,
    "num_experts": 4,
    "num_shared_expert": 1,
    "use_mixed_mlp_moe": True,
    "use_qk_norm": True,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "use_cla": True,
    "cla_share_factor": 2,
    "rope_scaling": {"type": "dynamic", "factor": 2.0, "alpha": 1.0},
    "tie_word_embeddings": False,
}


def test_hunyuan_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("hunyuan")
    args = ArgsCls.from_dict(MINIMAL_HUNYUAN)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
