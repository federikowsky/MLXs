"""Minimal forward test for RWKV7 (no weights, random init).

Cache design follows mlx_lm: one ArraysCache(size=3) per layer — [0] time-mixing
token shift, [1] WKV7 recurrence state (B, H, D, D), [2] channel-mixing token shift.
"""

import mlx.core as mx

from mlxs.load.registry import get_model_classes


def test_rwkv7_forward_minimal() -> None:
    """get_model_classes, from_dict(minimal), Model, make_cache, one forward → logits (B,T,V)."""
    ModelCls, ArgsCls = get_model_classes("rwkv7")
    minimal_config = {
        "model_type": "rwkv7",
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "norm_eps": 1e-5,
        "head_dim": 16,
        "num_hidden_layers": 2,
        "a_low_rank_dim": 4,
        "v_low_rank_dim": 4,
        "gate_low_rank_dim": 4,
        "decay_low_rank_dim": 4,
        "tie_word_embeddings": False,
    }
    args = ArgsCls.from_dict(minimal_config)
    model = ModelCls(args)
    cache = model.make_cache()
    assert len(cache) == args.num_hidden_layers
    assert all(len(c.cache) == 3 for c in cache)  # mlx_lm ArraysCache(size=3) per layer
    B, T = 1, 4
    input_ids = mx.array([[1, 2, 3, 4]])
    logits = model(input_ids, cache=cache)
    assert logits.shape == (B, T, args.vocab_size)
    assert model.num_layers == 2
    assert model.vocab_size == args.vocab_size
