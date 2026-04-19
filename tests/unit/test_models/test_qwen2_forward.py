"""qwen2: canonical text and multimodal family coverage."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes

MINIMAL_QWEN2 = {
    "model_type": "qwen2",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
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


def test_qwen2_text_forward() -> None:
    """Canonical qwen2 text path remains unchanged."""
    ModelCls, ArgsCls = get_model_classes("qwen2")
    args = ArgsCls.from_dict(MINIMAL_QWEN2)
    model = ModelCls(args)
    cache = model.make_cache()

    logits = model(mx.array([[1, 2, 3, 4]]), cache=cache)

    assert logits.shape == (1, 4, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
    assert model.supports_vision is False


def test_qwen2_multimodal_prepare_inputs() -> None:
    """Canonical qwen2 handles nested multimodal config via the unified family."""
    ModelCls, ArgsCls = get_model_classes("qwen2")
    args = ArgsCls.from_dict(
        {
            "model_type": "qwen2",
            "text_config": MINIMAL_QWEN2,
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
    assert input_embeddings.shape == (1, 4, MINIMAL_QWEN2["hidden_size"])
    assert logits.shape == (1, 4, model.vocab_size)


def test_qwen2_text_sanitize_uses_shared_family_remap() -> None:
    """qwen2 text sanitize keeps qwen2-specific cleanup on top of shared remap."""
    ModelCls, ArgsCls = get_model_classes("qwen2")
    args = ArgsCls.from_dict({**MINIMAL_QWEN2, "tie_word_embeddings": True})
    model = ModelCls(args)

    sanitized = model.sanitize(
        {
            "language_model.model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32),
            "visual.patch_embed.proj.weight": np.ones((2, 3, 1, 1, 1), dtype=np.float32),
            "mm_projector.weight": np.ones((2, 2), dtype=np.float32),
            "model.layers.0.self_attn.rotary_emb.inv_freq": np.ones((2,), dtype=np.float32),
            "lm_head.weight": np.ones((2, 2), dtype=np.float32),
        }
    )

    assert "model.embed_tokens.weight" in sanitized
    assert "language_model.model.embed_tokens.weight" not in sanitized
    assert "visual.patch_embed.proj.weight" not in sanitized
    assert "mm_projector.weight" not in sanitized
    assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in sanitized
    assert "lm_head.weight" not in sanitized


def test_qwen2_multimodal_sanitize_uses_shared_family_remap() -> None:
    """qwen2 multimodal sanitize preserves shared Qwen family remap semantics."""
    ModelCls, ArgsCls = get_model_classes("qwen2")
    args = ArgsCls.from_dict(
        {
            "model_type": "qwen2",
            "text_config": MINIMAL_QWEN2,
            "vision_config": MINIMAL_VISION_CONFIG,
            "image_token_id": 250,
        }
    )
    model = ModelCls(args, model_mode=ModelMode.MULTIMODAL)

    sanitized = model.sanitize(
        {
            "language_model.model.embed_tokens.weight": np.ones((2, 2), dtype=np.float32),
            "visual.patch_embed.proj.weight": np.ones((4, 3, 1, 1, 1), dtype=np.float32),
            "vision_model.blocks.0.norm1.weight": np.ones((16,), dtype=np.float32),
            "mm_projector.weight": np.ones((2, 2), dtype=np.float32),
        }
    )

    assert "model.embed_tokens.weight" in sanitized
    assert "vision_tower.patch_embed.proj.weight" in sanitized
    assert sanitized["vision_tower.patch_embed.proj.weight"].shape == (4, 1, 1, 1, 3)
    assert "vision_tower.blocks.0.norm1.weight" in sanitized
    assert "visual.patch_embed.proj.weight" not in sanitized
    assert "vision_model.blocks.0.norm1.weight" not in sanitized
    assert "mm_projector.weight" not in sanitized


def test_qwen2_flat_config_with_extra_metadata_keeps_top_level_text_args() -> None:
    """Flat checkpoints with extra metadata must not fall back to default text args."""
    ModelCls, ArgsCls = get_model_classes("qwen2")
    args = ArgsCls.from_dict(
        {
            **MINIMAL_QWEN2,
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_act": "silu",
            "use_cache": True,
        }
    )

    assert args.text_config  # exercises the flat-config leftover metadata path

    model = ModelCls(args)
    attn = model.layers[0].self_attn
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3, 4]]), cache=cache)

    assert attn.n_heads == MINIMAL_QWEN2["num_attention_heads"]
    assert attn.n_kv_heads == MINIMAL_QWEN2["num_key_value_heads"]
    assert attn.head_dim == MINIMAL_QWEN2["hidden_size"] // MINIMAL_QWEN2["num_attention_heads"]
    assert tuple(attn.q_proj.weight.shape) == (
        MINIMAL_QWEN2["hidden_size"],
        MINIMAL_QWEN2["hidden_size"],
    )
    assert tuple(attn.k_proj.weight.shape) == (
        MINIMAL_QWEN2["hidden_size"],
        MINIMAL_QWEN2["hidden_size"] // MINIMAL_QWEN2["num_attention_heads"] * MINIMAL_QWEN2["num_key_value_heads"],
    )
    assert logits.shape == (1, 4, args.vocab_size)
