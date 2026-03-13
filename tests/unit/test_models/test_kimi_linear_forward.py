"""Minimal forward test for kimi_linear (hybrid MLA + delta attention)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_KIMI_LINEAR = {
    "model_type": "kimi_linear",
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 128,
    "head_dim": 16,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "linear_attn_config": {
        "kda_layers": [2],
        "num_heads": 4,
        "head_dim": 16,
        "short_conv_kernel_size": 4,
    },
    "model_max_length": 128,
    "num_experts": 0,
    "moe_intermediate_size": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0,
    "v_head_dim": 16,
}


def test_kimi_linear_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("kimi_linear")
    args = ArgsCls.from_dict(MINIMAL_KIMI_LINEAR)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
