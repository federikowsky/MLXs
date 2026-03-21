"""Minimal forward test for Pixtral (text backbone only; no weights).

get_model_classes("pixtral"), ModelArgs.from_dict(minimal), Model(args),
make_cache(), forward → logits (B,T,V).
"""

from __future__ import annotations

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes

MINIMAL_PIXTRAL = {
    "model_type": "pixtral",
    "text_config": {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    },
}

MINIMAL_PIXTRAL_VISION = {
    "model_type": "pixtral",
    "num_hidden_layers": 1,
    "hidden_size": 16,
    "head_dim": 4,
    "intermediate_size": 32,
    "num_attention_heads": 4,
    "image_size": 4,
    "patch_size": 2,
    "projection_dim": 16,
    "num_channels": 3,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}


def test_pixtral_forward_minimal() -> None:
    """get_model_classes, from_dict, Model, make_cache(), forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("pixtral")
    args = ArgsCls.from_dict(MINIMAL_PIXTRAL)
    model = ModelCls(args)
    cache = model.make_cache()
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.text_config["vocab_size"])
    assert model.num_layers == 2
    assert model.vocab_size == 256
    assert model.supports_vision is False


def test_pixtral_prepare_inputs_multimodal() -> None:
    """Pixtral keeps family-specific encoder/projector while using the shared prepare skeleton."""
    ModelCls, ArgsCls = get_model_classes("pixtral")
    args = ArgsCls.from_dict(
        {
            **MINIMAL_PIXTRAL,
            "vision_config": MINIMAL_PIXTRAL_VISION,
            "image_token_id": 250,
        }
    )
    model = ModelCls(args, model_mode=ModelMode.MULTIMODAL)
    input_ids = mx.array([[250, 250, 250, 250]])

    prepared_ids, input_embeddings = model.prepare_inputs(
        input_ids,
        pixel_values=mx.zeros((1, 3, 4, 4), dtype=mx.float32),
        image_sizes=[(4, 4)],
    )
    logits = model(
        prepared_ids,
        cache=model.make_cache(),
        input_embeddings=input_embeddings,
    )

    assert model.supports_vision is True
    assert model.image_token_id == 250
    assert input_embeddings is not None
    assert input_embeddings.shape == (1, 4, 64)
    assert logits.shape == (1, 4, model.vocab_size)
