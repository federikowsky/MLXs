"""Longer end-to-end Adaptive KV stress coverage across retained runtime families."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlxs._types import GenerateOptions
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.generate import generate
from mlxs.models.llama import Model as LlamaModel
from mlxs.models.llama import ModelArgs as LlamaModelArgs
from mlxs.models.ministral3 import Model as Ministral3Model
from mlxs.models.ministral3 import ModelArgs as Ministral3ModelArgs
from mlxs.models.qwen3_5 import Model as Qwen35Model
from mlxs.models.qwen3_5 import ModelArgs as Qwen35ModelArgs
from mlxs.observability.metrics import InMemoryMetrics


class _NoStopTokenizer:
    eos_token_id = 999_999

    def encode(self, text: str) -> list[int]:
        del text
        return [1, 2, 3, 4]

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return chr(ord("a") + token_ids % 26)
        return "".join(chr(ord("a") + token % 26) for token in token_ids)


def _tokens() -> list[int]:
    return [((i % 96) + 1) for i in range(128)]


def _long_tokens(n: int = 1024) -> list[int]:
    return [((i % 96) + 1) for i in range(n)]


def test_llama_long_chunked_prefill_hard_recovery_preserves_generation() -> None:
    mx.random.seed(101)
    tokenizer = _NoStopTokenizer()
    model = LlamaModel(
        LlamaModelArgs(
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=128,
            head_dim=32,
            layer_types=["full_attention"] * 4,
        )
    )

    prompt = _tokens()
    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            prefill_step_size=16,
        )
    ]

    metrics = InMemoryMetrics()
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=8,
                update_window_steps=4,
                soft_budget_bytes=12000,
                hard_budget_bytes=18000,
                recent_tail_protect_blocks=0,
            ),
            prefill_step_size=16,
            metrics=metrics,
        )
    ]

    assert adaptive == baseline
    assert metrics.get_counter("adaptive_kv_evictions_total") > 0
    assert metrics.get_counter("adaptive_kv_recomputations_total") > 0


def test_standard_ministral3_long_chunked_prefill_hard_recovery_preserves_generation() -> None:
    mx.random.seed(103)
    tokenizer = _NoStopTokenizer()
    model = Ministral3Model(
        Ministral3ModelArgs(
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            vocab_size=128,
            tie_word_embeddings=False,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            sliding_window=64,
        )
    )

    prompt = _tokens()
    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            prefill_step_size=16,
        )
    ]

    metrics = InMemoryMetrics()
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=8,
                update_window_steps=4,
                soft_budget_bytes=12000,
                hard_budget_bytes=18000,
                recent_tail_protect_blocks=0,
            ),
            prefill_step_size=16,
            metrics=metrics,
        )
    ]

    assert adaptive == baseline
    assert metrics.get_counter("adaptive_kv_evictions_total") > 0
    assert metrics.get_counter("adaptive_kv_recomputations_total") > 0


def test_standard_ministral3_chunked_prefill_matches_unchunked_and_adaptive_full() -> None:
    mx.random.seed(109)
    tokenizer = _NoStopTokenizer()
    model_args = Ministral3ModelArgs(
        hidden_size=128,
        num_hidden_layers=4,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        vocab_size=128,
        tie_word_embeddings=False,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        sliding_window=64,
    )
    prompt = [((i % 96) + 1) for i in range(256)]

    mx.random.seed(109)
    unchunked_model = Ministral3Model(model_args)
    baseline = [
        event.token_id
        for event in generate(
            unchunked_model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            prefill_step_size=512,
        )
    ]

    mx.random.seed(109)
    chunked_model = Ministral3Model(model_args)
    chunked = [
        event.token_id
        for event in generate(
            chunked_model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            prefill_step_size=16,
        )
    ]

    mx.random.seed(109)
    adaptive_model = Ministral3Model(model_args)
    adaptive = [
        event.token_id
        for event in generate(
            adaptive_model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            prefill_step_size=16,
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=8,
                update_window_steps=4,
                soft_budget_bytes=10**9,
                hard_budget_bytes=10**9,
            ),
        )
    ]

    assert chunked == baseline
    assert adaptive == baseline


def test_standard_qwen35_long_chunked_prefill_hard_recovery_preserves_generation() -> None:
    mx.random.seed(107)
    tokenizer = _NoStopTokenizer()
    model = Qwen35Model(
        Qwen35ModelArgs(
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            full_attention_interval=2,
            vocab_size=128,
            tie_word_embeddings=False,
        )
    )

    prompt = _tokens()
    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            prefill_step_size=16,
        )
    ]

    metrics = InMemoryMetrics()
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=16, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=8,
                update_window_steps=4,
                soft_budget_bytes=12000,
                hard_budget_bytes=18000,
                recent_tail_protect_blocks=0,
            ),
            prefill_step_size=16,
            metrics=metrics,
        )
    ]

    assert adaptive == baseline
    assert metrics.get_counter("adaptive_kv_evictions_total") > 0
    assert metrics.get_counter("adaptive_kv_recomputations_total") > 0


@pytest.mark.parametrize(
    ("pressure_mode", "soft_budget_bytes", "hard_budget_bytes"),
    [
        ("soft", 30_000, 30_000),
        ("hard", 12_000, 18_000),
    ],
)
def test_standard_ministral3_long_resident_profiles_preserve_generation(
    pressure_mode: str,
    soft_budget_bytes: int,
    hard_budget_bytes: int,
) -> None:
    mx.random.seed(123)
    tokenizer = _NoStopTokenizer()
    prompt = _long_tokens()
    model_args = Ministral3ModelArgs(
        hidden_size=128,
        num_hidden_layers=4,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        vocab_size=128,
        tie_word_embeddings=False,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        sliding_window=64,
    )
    baseline = [
        event.token_id
        for event in generate(
            Ministral3Model(model_args),
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=96, temperature=0),
            prefill_step_size=16,
        )
    ]

    mx.random.seed(123)
    final_state: list[dict[str, object]] = []
    adaptive = [
        event.token_id
        for event in generate(
            Ministral3Model(model_args),
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=96, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=8,
                update_window_steps=4,
                soft_budget_bytes=soft_budget_bytes,
                hard_budget_bytes=hard_budget_bytes,
                recent_tail_protect_blocks=0,
            ),
            prefill_step_size=16,
            final_adaptive_state_out=final_state,
        )
    ]

    assert adaptive == baseline
    assert final_state
    blocks = final_state[0]["blocks"]
    assert any(block["profile"] == "tq_aggr" for block in blocks)
    if pressure_mode == "hard":
        assert final_state[0]["pressure_state"] in {"soft", "hard"}


@pytest.mark.parametrize(
    ("pressure_mode", "soft_budget_bytes", "hard_budget_bytes"),
    [
        ("soft", 30_000, 30_000),
        ("hard", 12_000, 18_000),
    ],
)
def test_standard_qwen35_long_resident_profiles_preserve_generation(
    pressure_mode: str,
    soft_budget_bytes: int,
    hard_budget_bytes: int,
) -> None:
    mx.random.seed(123)
    tokenizer = _NoStopTokenizer()
    prompt = _long_tokens()
    model_args = Qwen35ModelArgs(
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=2,
        vocab_size=128,
        tie_word_embeddings=False,
    )
    baseline = [
        event.token_id
        for event in generate(
            Qwen35Model(model_args),
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=96, temperature=0),
            prefill_step_size=16,
        )
    ]

    mx.random.seed(123)
    final_state: list[dict[str, object]] = []
    adaptive = [
        event.token_id
        for event in generate(
            Qwen35Model(model_args),
            tokenizer,
            prompt,
            GenerateOptions(max_tokens=96, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=8,
                update_window_steps=4,
                soft_budget_bytes=soft_budget_bytes,
                hard_budget_bytes=hard_budget_bytes,
                recent_tail_protect_blocks=0,
            ),
            prefill_step_size=16,
            final_adaptive_state_out=final_state,
        )
    ]

    assert adaptive == baseline
    assert final_state
    blocks = final_state[0]["blocks"]
    assert any(block["profile"] == "tq_aggr" for block in blocks)
    if pressure_mode == "hard":
        assert final_state[0]["pressure_state"] in {"soft", "hard"}
