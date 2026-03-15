"""qwen3_5_vl: registry lookup, text forward, and multimodal input preparation."""

from __future__ import annotations

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes

MINIMAL_QWEN3_5_VL = {
    "model_type": "qwen3_5_vl",
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

MINIMAL_VISION_CONFIG = {
    "depth": 1,
    "hidden_size": 16,
    "out_hidden_size": 64,
    "num_heads": 4,
    "patch_size": 1,
    "temporal_patch_size": 1,
    "spatial_merge_size": 1,
    "in_channels": 3,
    "mlp_ratio": 2.0,
}


def test_qwen3_5_vl_forward() -> None:
    """Registry + flat config works in text mode."""
    ModelCls, ArgsCls = get_model_classes("qwen3_5_vl")
    args = ArgsCls.from_dict(MINIMAL_QWEN3_5_VL)
    model = ModelCls(args)
    cache = model.make_cache()

    logits = model(mx.array([[1, 2, 3, 4]]), cache=cache)

    assert logits.shape == (1, 4, model.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
    assert model.supports_vision is False


def test_qwen3_5_vl_registry_uses_unified_qwen3_5_classes() -> None:
    """Legacy qwen3_5_vl registry key resolves to the unified qwen3_5 implementation."""
    vl_model_cls, vl_args_cls = get_model_classes("qwen3_5_vl")
    qwen_model_cls, qwen_args_cls = get_model_classes("qwen3_5")

    assert vl_model_cls is qwen_model_cls
    assert vl_args_cls is qwen_args_cls


def test_qwen3_5_vl_prepare_inputs_multimodal() -> None:
    """MULTIMODAL mode merges image features into the text embedding stream."""
    ModelCls, ArgsCls = get_model_classes("qwen3_5_vl")
    args = ArgsCls.from_dict(
        {
            **MINIMAL_QWEN3_5_VL,
            "vision_config": MINIMAL_VISION_CONFIG,
        }
    )
    model = ModelCls(args, model_mode=ModelMode.MULTIMODAL)
    input_ids = mx.array([[1, args.image_token_id, 2, 3]])
    pixel_values = mx.zeros((1, 3, 1, 1), dtype=mx.float32)
    image_grid_thw = mx.array([[1, 1, 1]])

    prepared_ids, input_embeddings = model.prepare_inputs(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    logits = model(
        prepared_ids,
        cache=model.make_cache(),
        input_embeddings=input_embeddings,
    )

    assert model.supports_vision is True
    assert model.image_token_id == args.image_token_id
    assert input_embeddings is not None
    assert input_embeddings.shape == (1, 4, MINIMAL_QWEN3_5_VL["hidden_size"])
    assert logits.shape == (1, 4, model.vocab_size)


def test_qwen3_5_vl_prepare_inputs_video_path() -> None:
    """Legacy qwen3_5_vl key still supports the video-specific prepare path."""
    ModelCls, ArgsCls = get_model_classes("qwen3_5_vl")
    args = ArgsCls.from_dict(
        {
            **MINIMAL_QWEN3_5_VL,
            "vision_config": MINIMAL_VISION_CONFIG,
        }
    )
    model = ModelCls(args, model_mode=ModelMode.MULTIMODAL)
    input_ids = mx.array([[1, args.video_token_id, 2, 3]])

    prepared_ids, input_embeddings = model.prepare_inputs(
        input_ids,
        video_pixel_values=mx.zeros((1, 3, 1, 1), dtype=mx.float32),
        video_grid_thw=mx.array([[1, 1, 1]]),
    )

    assert mx.array_equal(prepared_ids, input_ids)
    assert input_embeddings is not None
    assert input_embeddings.shape == (1, 4, MINIMAL_QWEN3_5_VL["hidden_size"])
