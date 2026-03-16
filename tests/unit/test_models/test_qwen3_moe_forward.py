"""qwen3_moe: canonical text and multimodal family coverage."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes

MINIMAL_QWEN3_MOE: dict[str, Any] = {
    "model_type": "qwen3_moe",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "moe_intermediate_size": 64,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "max_position_embeddings": 256,
    "norm_topk_prob": True,
    "rope_theta": 10000.0,
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


def test_qwen3_moe_text_forward() -> None:
    """Canonical qwen3_moe text path remains unchanged."""
    ModelCls, ArgsCls = get_model_classes("qwen3_moe")
    args = ArgsCls.from_dict(MINIMAL_QWEN3_MOE)
    model = ModelCls(args)

    logits = model(mx.array([[1, 2, 3, 4]]), cache=model.make_cache())

    assert logits.shape == (1, 4, model.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == 256
    assert model.supports_vision is False


def test_qwen3_moe_multimodal_prepare_inputs() -> None:
    """Canonical qwen3_moe handles nested multimodal config via the unified family."""
    ModelCls, ArgsCls = get_model_classes("qwen3_moe")
    args = ArgsCls.from_dict(
        {
            "model_type": "qwen3_moe",
            "text_config": MINIMAL_QWEN3_MOE,
            "vision_config": MINIMAL_VISION_CONFIG,
            "image_token_id": 250,
        }
    )
    model = ModelCls(args, model_mode=ModelMode.MULTIMODAL)
    input_ids = mx.array([[1, args.image_token_id, 2, 3]])

    prepared_ids, input_embeddings = model.prepare_inputs(
        input_ids,
        pixel_values=mx.zeros((1, 3, 1, 1), dtype=mx.float32),
        image_grid_thw=mx.array([[1, 1, 1]]),
    )
    logits = model(
        prepared_ids,
        cache=model.make_cache(),
        input_embeddings=input_embeddings,
    )

    assert model.supports_vision is True
    assert model.image_token_id == args.image_token_id
    assert input_embeddings is not None
    assert input_embeddings.shape == (1, 4, MINIMAL_QWEN3_MOE["hidden_size"])
    assert logits.shape == (1, 4, model.vocab_size)
