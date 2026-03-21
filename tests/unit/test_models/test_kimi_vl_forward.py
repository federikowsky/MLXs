"""kimi_vl: text and multimodal wrapper coverage."""

from __future__ import annotations

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes

TEXT_HIDDEN_SIZE = 64

MINIMAL_KIMI_VL = {
    "model_type": "kimi_vl",
    "text_config": {
        "model_type": "deepseek_v3",
        "hidden_size": TEXT_HIDDEN_SIZE,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 16,
        "v_head_dim": 32,
        "kv_lora_rank": 32,
        "q_lora_rank": 32,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "n_routed_experts": 4,
        "n_shared_experts": None,
        "num_experts_per_tok": 1,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 0,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "rope_theta": 10000.0,
    },
}


MINIMAL_KIMI_VL_VISION = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "patch_size": 2,
    "image_size": 4,
    "spatial_merge_size": 1,
    "merge_kernel_size": [1, 1],
}


def test_kimi_vl_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("kimi_vl")
    args = ArgsCls.from_dict(MINIMAL_KIMI_VL)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    V = args.text_config.vocab_size
    assert logits.shape == (B, T, V)
    assert model.num_layers == 2
    assert model.vocab_size == V


def test_kimi_vl_multimodal_prepare_inputs_with_public_aliases() -> None:
    """kimi_vl accepts public config aliases and real multimodal prepare_inputs."""
    ModelCls, ArgsCls = get_model_classes("kimi_vl")
    args = ArgsCls.from_dict(
        {
            "model_type": "kimi_vl",
            "text_config": MINIMAL_KIMI_VL["text_config"],
            "vision_config": MINIMAL_KIMI_VL_VISION,
            "media_placeholder_token_id": 250,
        }
    )
    model = ModelCls(args, model_mode=ModelMode.MULTIMODAL)
    input_ids = mx.array([[250, 250, 250, 250]], dtype=mx.int32)

    prepared_ids, input_embeddings = model.prepare_inputs(
        input_ids,
        pixel_values=mx.zeros((4, 2, 2, 3), dtype=mx.float32),
        image_grid_thw=mx.array([[2, 2]], dtype=mx.int32),
    )
    logits = model(
        prepared_ids,
        cache=model.make_cache(),
        input_embeddings=input_embeddings,
    )

    assert model.supports_vision is True
    assert model.image_token_id == 250
    assert input_embeddings is not None
    assert input_embeddings.shape == (1, 4, TEXT_HIDDEN_SIZE)
    assert logits.shape == (1, 4, model.vocab_size)
