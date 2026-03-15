"""lfm2/lfm2_vl: text, multimodal, and registry compatibility."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes

MINIMAL_LFM2_VL: dict[str, Any] = {
    "model_type": "lfm2_vl",
    "text_config": {
        "model_type": "lfm2",
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
        "norm_eps": 1e-6,
        "conv_bias": False,
        "conv_L_cache": 4,
        "block_dim": 64,
        "block_ff_dim": 128,
        "block_multiple_of": 64,
        "block_ffn_dim_multiplier": None,
        "block_auto_adjust_ff_dim": True,
        "rope_theta": 10000.0,
        "layer_types": ["full_attention", "short_conv"],
    },
}

MINIMAL_LFM2_VISION: dict[str, Any] = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_channels": 3,
    "image_size": 4,
    "patch_size": 2,
    "num_patches": 4,
}


def test_lfm2_vl_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("lfm2_vl")
    args = ArgsCls.from_dict(MINIMAL_LFM2_VL)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.text_config["vocab_size"])
    assert model.num_layers == 2
    assert model.vocab_size == 256
    assert model.supports_vision is False


def test_lfm2_vl_registry_uses_unified_lfm2_classes() -> None:
    """Legacy lfm2_vl key resolves to the unified lfm2 implementation."""
    vl_model_cls, vl_args_cls = get_model_classes("lfm2_vl")
    lfm2_model_cls, lfm2_args_cls = get_model_classes("lfm2")

    assert vl_model_cls is lfm2_model_cls
    assert vl_args_cls is lfm2_args_cls


def test_lfm2_vl_prepare_inputs_multimodal() -> None:
    """Legacy lfm2_vl key still exercises the unified multimodal path."""
    ModelCls, ArgsCls = get_model_classes("lfm2_vl")
    args = ArgsCls.from_dict(
        {
            **MINIMAL_LFM2_VL,
            "vision_config": MINIMAL_LFM2_VISION,
            "image_token_id": 250,
        }
    )
    model = ModelCls(args, model_mode=ModelMode.MULTIMODAL)
    input_ids = mx.array([[250, 250, 250, 250]])

    prepared_ids, input_embeddings = model.prepare_inputs(
        input_ids,
        pixel_values=mx.zeros((1, 3, 4, 4), dtype=mx.float32),
    )

    assert mx.array_equal(prepared_ids, input_ids)
    assert input_embeddings is not None
    assert input_embeddings.shape == (1, 4, MINIMAL_LFM2_VL["text_config"]["hidden_size"])
