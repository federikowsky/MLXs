"""Minimal forward test for lfm2 (short conv + full attention)."""

from __future__ import annotations

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes

MINIMAL_LFM2 = {
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
}

MINIMAL_LFM2_VISION = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_channels": 3,
    "image_size": 4,
    "patch_size": 2,
    "num_patches": 4,
}


def test_lfm2_forward() -> None:
    """From registry + minimal config: Model, make_cache, forward -> logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("lfm2")
    args = ArgsCls.from_dict(MINIMAL_LFM2)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256


def test_lfm2_multimodal_prepare_inputs() -> None:
    """Unified lfm2 handles nested multimodal config and 4D image tensors."""
    ModelCls, ArgsCls = get_model_classes("lfm2")
    args = ArgsCls.from_dict(
        {
            "model_type": "lfm2",
            "text_config": MINIMAL_LFM2,
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
    logits = model(
        prepared_ids,
        cache=model.make_cache(),
        input_embeddings=input_embeddings,
    )

    assert model.supports_vision is True
    assert model.image_token_id == args.image_token_id
    assert input_embeddings is not None
    assert input_embeddings.shape == (1, 4, MINIMAL_LFM2["hidden_size"])
    assert logits.shape == (1, 4, model.vocab_size)
