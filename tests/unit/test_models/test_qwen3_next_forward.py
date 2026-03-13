"""qwen3_next: registry, from_dict(minimal), make_cache(), one forward -> logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_QWEN3_NEXT = {
    "model_type": "qwen3_next",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "max_position_embeddings": 256,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "num_experts": 0,
    "num_experts_per_tok": 0,
    "decoder_sparse_step": 1,
    "shared_expert_intermediate_size": 128,
    "mlp_only_layers": [],
    "moe_intermediate_size": 128,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "partial_rotary_factor": 0.25,
}


def test_qwen3_next_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("qwen3_next")
    args = ArgsCls.from_dict(MINIMAL_QWEN3_NEXT)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
