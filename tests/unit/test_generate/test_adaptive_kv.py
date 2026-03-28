"""Adaptive KV integration tests for single-request generation."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from mlxs._types import GenerateOptions
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.cache.kv import KVCache
from mlxs.generate import generate
from mlxs.layers.attention import scaled_dot_product_attention
from mlxs.models.llama import Model as LlamaModel
from mlxs.models.llama import ModelArgs as LlamaModelArgs
from mlxs.models.ministral3 import Model as Ministral3Model
from mlxs.models.ministral3 import ModelArgs as Ministral3ModelArgs
from mlxs.models.qwen3_5 import Model as Qwen35Model
from mlxs.models.qwen3_5 import ModelArgs as Qwen35ModelArgs
from mlxs.observability.metrics import InMemoryMetrics


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3, 4]

    def decode(self, token_ids: int | list[int]) -> str:
        if isinstance(token_ids, int):
            return chr(ord("a") + token_ids % 26)
        return "".join(chr(ord("a") + token % 26) for token in token_ids)


class _AdaptiveBaselineModel:
    model_type = "llama"

    def __init__(self, vocab_size: int = 16, head_dim: int = 32) -> None:
        self._vocab_size = vocab_size
        self._layers = 1
        self._head_dim = head_dim

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[Any] | None = None,
        mask: mx.array | str | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        del input_embeddings
        batch, seq = input_ids.shape
        base = input_ids.astype(mx.float32).reshape(batch, 1, seq, 1)
        q = mx.broadcast_to(base, (batch, 1, seq, self._head_dim))
        k = mx.broadcast_to(base + 1, (batch, 1, seq, self._head_dim))
        v = mx.ones((batch, 1, seq, self._head_dim), dtype=mx.float32)
        if cache is not None:
            k, v = cache[0].update_and_fetch(k, v)
            _ = scaled_dot_product_attention(q, k, v, cache=cache[0], scale=1.0, mask=mask)
        logits = mx.zeros((batch, seq, self._vocab_size), dtype=mx.float32)
        next_token = int((input_ids[0, -1].item() + 1) % self._vocab_size)
        logits[:, :, next_token] = 10.0
        return logits

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in range(self._layers)]

    @property
    def num_layers(self) -> int:
        return self._layers

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class _UnsupportedModel(_AdaptiveBaselineModel):
    model_type = "qwen"


class _NoStopTokenizer(_Tokenizer):
    eos_token_id = 999_999


def test_adaptive_disabled_preserves_existing_behavior() -> None:
    tokenizer = _Tokenizer()
    model = _AdaptiveBaselineModel()
    events = list(
        generate(
            model,
            tokenizer,
            [1, 2, 3],
            GenerateOptions(max_tokens=2, temperature=0),
        )
    )

    assert len(events) == 2
    assert events[0].token_id == 4


def test_adaptive_enabled_supported_baseline_emits_snapshot_and_metrics() -> None:
    tokenizer = _Tokenizer()
    model = _AdaptiveBaselineModel()
    metrics = InMemoryMetrics()
    final_state: list[dict[str, Any]] = []

    events = list(
        generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=2,
                update_window_steps=1,
                soft_budget_bytes=1,
                hard_budget_bytes=1,
                t_tq_safe_restore=0.95,
                t_tq_safe_degrade=0.9,
                t_evict_candidate=0.99,
            ),
            metrics=metrics,
            final_adaptive_state_out=final_state,
        )
    )

    assert len(events) == 4
    assert final_state
    snapshot = final_state[0]
    assert "blocks" in snapshot
    assert metrics.get_counter("adaptive_kv_score_updates_total") > 0
    assert metrics.get_gauge("adaptive_kv_blocks_total") >= 1


def test_adaptive_enabled_unsupported_family_fails_clearly() -> None:
    tokenizer = _Tokenizer()
    model = _UnsupportedModel()

    with pytest.raises(Exception, match="no registered exact adapter"):
        list(
            generate(
                model,
                tokenizer,
                [1, 2, 3],
                GenerateOptions(max_tokens=1, temperature=0),
                adaptive_config=AdaptiveKVConfig(enabled=True),
            )
        )


def test_adaptive_enabled_standard_qwen35_hybrid_runtime_preserves_generation() -> None:
    mx.random.seed(19)
    tokenizer = _NoStopTokenizer()
    model = Qwen35Model(
        Qwen35ModelArgs(
            hidden_size=128,
            num_hidden_layers=2,
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

    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
        )
    ]

    metrics = InMemoryMetrics()
    final_state: list[dict[str, Any]] = []
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=2,
                update_window_steps=1,
                soft_budget_bytes=800,
                hard_budget_bytes=1200,
                recent_tail_protect_blocks=0,
                t_tq_safe_restore=0.95,
                t_tq_safe_degrade=0.9,
                t_evict_candidate=0.99,
            ),
            metrics=metrics,
            final_adaptive_state_out=final_state,
        )
    ]

    assert adaptive == baseline
    assert final_state
    assert "n_execution_slabs" in final_state[0]["attention_path"]
    assert "n_visible_slices" in final_state[0]["attention_path"]
    assert metrics.get_counter("adaptive_kv_evictions_total") > 0
    assert metrics.get_counter("adaptive_kv_recomputations_total") > 0


def test_adaptive_enabled_standard_ministral3_sliding_runtime_preserves_generation() -> None:
    mx.random.seed(23)
    tokenizer = _NoStopTokenizer()
    model = Ministral3Model(
        Ministral3ModelArgs(
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            vocab_size=128,
            tie_word_embeddings=False,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=64,
        )
    )

    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
        )
    ]

    metrics = InMemoryMetrics()
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=2,
                update_window_steps=1,
                soft_budget_bytes=800,
                hard_budget_bytes=1200,
                recent_tail_protect_blocks=0,
                t_tq_safe_restore=0.95,
                t_tq_safe_degrade=0.9,
                t_evict_candidate=0.99,
            ),
            metrics=metrics,
        )
    ]

    assert adaptive == baseline
    assert metrics.get_counter("adaptive_kv_evictions_total") > 0
    assert metrics.get_counter("adaptive_kv_recomputations_total") > 0


def test_adaptive_enabled_rejects_compile_decode() -> None:
    tokenizer = _Tokenizer()
    model = _AdaptiveBaselineModel()

    with pytest.raises(Exception, match="compile_decode=True"):
        list(
            generate(
                model,
                tokenizer,
                [1, 2, 3],
                GenerateOptions(max_tokens=1, temperature=0),
                compile_decode=True,
                adaptive_config=AdaptiveKVConfig(enabled=True),
            )
        )


def test_adaptive_real_llama_recovery_preserves_generation_under_pressure() -> None:
    mx.random.seed(7)
    tokenizer = _NoStopTokenizer()
    model = LlamaModel(
        LlamaModelArgs(
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
            head_dim=32,
            layer_types=["full_attention", "full_attention"],
        )
    )

    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
        )
    ]

    metrics = InMemoryMetrics()
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=2,
                update_window_steps=1,
                soft_budget_bytes=500,
                hard_budget_bytes=900,
                recent_tail_protect_blocks=0,
                t_tq_safe_restore=0.95,
                t_tq_safe_degrade=0.9,
                t_evict_candidate=0.99,
            ),
            metrics=metrics,
        )
    ]

    assert adaptive == baseline
    assert metrics.get_counter("adaptive_kv_evictions_total") > 0
    assert metrics.get_counter("adaptive_kv_recomputations_total") > 0


def test_adaptive_qwen35_full_attention_only_recovery_preserves_generation() -> None:
    mx.random.seed(11)
    tokenizer = _NoStopTokenizer()
    model = Qwen35Model(
        Qwen35ModelArgs(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=256,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            full_attention_interval=1,
            vocab_size=128,
            tie_word_embeddings=False,
        )
    )

    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
        )
    ]

    metrics = InMemoryMetrics()
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=2,
                update_window_steps=1,
                soft_budget_bytes=800,
                hard_budget_bytes=1200,
                recent_tail_protect_blocks=0,
                t_tq_safe_restore=0.95,
                t_tq_safe_degrade=0.9,
                t_evict_candidate=0.99,
            ),
            metrics=metrics,
        )
    ]

    assert adaptive == baseline
    assert metrics.get_counter("adaptive_kv_evictions_total") > 0
    assert metrics.get_counter("adaptive_kv_recomputations_total") > 0


def test_adaptive_ministral3_full_attention_only_recovery_preserves_generation() -> None:
    mx.random.seed(13)
    tokenizer = _NoStopTokenizer()
    model = Ministral3Model(
        Ministral3ModelArgs(
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            vocab_size=128,
            tie_word_embeddings=False,
            layer_types=["full_attention", "full_attention"],
            sliding_window=None,
        )
    )

    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
        )
    ]

    metrics = InMemoryMetrics()
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4],
            GenerateOptions(max_tokens=4, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=2,
                update_window_steps=1,
                soft_budget_bytes=800,
                hard_budget_bytes=1200,
                recent_tail_protect_blocks=0,
                t_tq_safe_restore=0.95,
                t_tq_safe_degrade=0.9,
                t_evict_candidate=0.99,
            ),
            metrics=metrics,
        )
    ]

    assert adaptive == baseline
    assert metrics.get_counter("adaptive_kv_evictions_total") > 0
    assert metrics.get_counter("adaptive_kv_recomputations_total") > 0


def test_adaptive_hard_stabilization_preserves_parity_and_avoids_repeat_churn() -> None:
    tokenizer = _NoStopTokenizer()
    model = _AdaptiveBaselineModel()

    baseline = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4, 5, 6],
            GenerateOptions(max_tokens=8, temperature=0),
        )
    ]

    metrics = InMemoryMetrics()
    final_state: list[dict[str, Any]] = []
    adaptive = [
        event.token_id
        for event in generate(
            model,
            tokenizer,
            [1, 2, 3, 4, 5, 6],
            GenerateOptions(max_tokens=8, temperature=0),
            adaptive_config=AdaptiveKVConfig(
                enabled=True,
                block_size_tokens=2,
                update_window_steps=1,
                soft_budget_bytes=1,
                hard_budget_bytes=1,
                recent_tail_protect_blocks=0,
                t_tq_safe_restore=0.95,
                t_tq_safe_degrade=0.9,
                t_evict_candidate=0.99,
            ),
            metrics=metrics,
            final_adaptive_state_out=final_state,
        )
    ]

    assert adaptive == baseline
    assert metrics.get_counter("adaptive_kv_recomputations_total") == 1
    assert final_state
    stabilization = final_state[0]["hard_stabilization"]
    assert final_state[0]["pressure_state"] == "hard"
    assert stabilization["episode_active"] is True
    assert stabilization["recovery_hold_active"] is True
    assert stabilization["best_achievable_under_current_forward_semantics"] is True
    assert stabilization["reason"] == "required_history_recovered_under_hard_episode"
    assert stabilization["stabilized_block_ids"] == [1, 2]
    assert metrics.get_counter("adaptive_kv_evictions_total") == len(
        stabilization["stabilized_block_ids"]
    )
