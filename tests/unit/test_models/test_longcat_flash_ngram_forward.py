"""Minimal forward test for longcat_flash_ngram (ngram embedding + longcat_flash layers)."""

from __future__ import annotations

import mlx.core as mx

from mlxs.load.registry import get_model_classes

MINIMAL_LONGCAT_FLASH_NGRAM = {
    "model_type": "longcat_flash_ngram",
    "attention_method": "flash",
    "zero_expert_type": "identity",
    "hidden_size": 64,
    "ffn_hidden_size": 128,
    "moe_topk": 1,
    "expert_ffn_hidden_size": 64,
    "n_routed_experts": 2,
    "zero_expert_num": 1,
    "num_layers": 2,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "num_attention_heads": 4,
    "kv_lora_rank": 32,
    "q_lora_rank": 32,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 16,
    "v_head_dim": 32,
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "mla_scale_q_lora": False,
    "mla_scale_kv_lora": False,
    "attention_bias": False,
    "norm_topk_prob": False,
    "router_bias": False,
    "ngram_vocab_size_ratio": 78,
    "emb_neighbor_num": 4,
    "emb_split_num": 4,
}


def test_longcat_flash_ngram_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("longcat_flash_ngram")
    args = ArgsCls.from_dict(MINIMAL_LONGCAT_FLASH_NGRAM)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
