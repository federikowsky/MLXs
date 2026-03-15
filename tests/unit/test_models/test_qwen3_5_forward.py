"""Minimal forward test for qwen3_5: registry, from_dict, make_cache, forward -> logits (B,T,V)."""

from __future__ import annotations

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes

MINIMAL_QWEN3_5 = {
    "model_type": "qwen3_5",
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
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "partial_rotary_factor": 0.25,
}


def test_qwen3_5_forward() -> None:
    """Registry, from_dict(minimal), Model, make_cache(), forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("qwen3_5")
    args = ArgsCls.from_dict(MINIMAL_QWEN3_5)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256


def test_qwen3_5_forward_with_input_embeddings() -> None:
    """Precomputed embeddings can replace token embedding lookup."""
    ModelCls, ArgsCls = get_model_classes("qwen3_5")
    args = ArgsCls.from_dict(MINIMAL_QWEN3_5)
    model = ModelCls(args)
    cache = model.make_cache()
    input_ids = mx.array([[1, 2, 3, 4]])
    input_embeddings = mx.zeros((1, 4, args.hidden_size), dtype=mx.float32)

    logits = model(input_ids, cache=cache, input_embeddings=input_embeddings)

    assert logits.shape == (1, 4, args.vocab_size)


def test_qwen3_5_multimodal_prepare_inputs_with_video() -> None:
    """Unified qwen3_5 handles nested multimodal config and video placeholders."""
    ModelCls, ArgsCls = get_model_classes("qwen3_5")
    args = ArgsCls.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": MINIMAL_QWEN3_5,
            "vision_config": {
                "depth": 1,
                "hidden_size": 16,
                "out_hidden_size": 64,
                "num_heads": 4,
                "patch_size": 1,
                "temporal_patch_size": 1,
                "spatial_merge_size": 1,
                "in_channels": 3,
                "mlp_ratio": 2.0,
            },
            "image_token_id": 250,
            "video_token_id": 251,
        }
    )
    model = ModelCls(args, model_mode=ModelMode.MULTIMODAL)
    input_ids = mx.array([[1, args.image_token_id, args.video_token_id, 2]])

    prepared_ids, input_embeddings = model.prepare_inputs(
        input_ids,
        pixel_values=mx.zeros((1, 3, 1, 1), dtype=mx.float32),
        image_grid_thw=mx.array([[1, 1, 1]]),
        video_pixel_values=mx.zeros((1, 3, 1, 1), dtype=mx.float32),
        video_grid_thw=mx.array([[1, 1, 1]]),
    )
    logits = model(
        prepared_ids,
        cache=model.make_cache(),
        input_embeddings=input_embeddings,
    )

    assert model.supports_vision is True
    assert model.image_token_id == args.image_token_id
    assert input_embeddings is not None
    assert input_embeddings.shape == (1, 4, MINIMAL_QWEN3_5["hidden_size"])
    assert logits.shape == (1, 4, model.vocab_size)
