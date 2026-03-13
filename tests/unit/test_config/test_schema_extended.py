"""Extended config schema tests — boundary, corner cases, alternate flows (§8, AC3).

Supplements test_schema.py with deeper coverage.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlxs._types import MemoryCeilingPolicy, ModelMode, PaddingSide, StreamPolicy
from mlxs.config.schema import (
    AppConfig,
    BatchConfig,
    CacheConfig,
    GenerateConfig,
    ModelConfig,
    PromptCacheConfig,
    ServerConfig,
    SpeculativeConfig,
    ToolCallingConfig,
)

# =============================================================================
# ModelConfig
# =============================================================================


class TestModelConfigAlternateFlows:
    def test_model_mode_text(self) -> None:
        cfg = ModelConfig(model_mode=ModelMode.TEXT)
        assert cfg.model_mode == ModelMode.TEXT

    def test_model_mode_multimodal(self) -> None:
        cfg = ModelConfig(model_mode=ModelMode.MULTIMODAL)
        assert cfg.model_mode == ModelMode.MULTIMODAL

    def test_model_mode_auto_default(self) -> None:
        cfg = ModelConfig()
        assert cfg.model_mode == ModelMode.AUTO

    def test_preload_overrides_lazy(self) -> None:
        """Both can be set; semantics are in the loader, not in validation."""
        cfg = ModelConfig(lazy_load=True, preload=True)
        assert cfg.preload is True
        assert cfg.lazy_load is True


class TestModelConfigBoundary:
    def test_empty_model_path(self) -> None:
        cfg = ModelConfig()
        assert cfg.model_path == ""

    def test_adapter_path_none(self) -> None:
        cfg = ModelConfig()
        assert cfg.adapter_path is None


# =============================================================================
# GenerateConfig
# =============================================================================


class TestGenerateConfigBoundary:
    def test_max_tokens_zero(self) -> None:
        cfg = GenerateConfig(max_tokens=0)
        assert cfg.max_tokens == 0

    def test_temperature_zero(self) -> None:
        cfg = GenerateConfig(temperature=0.0)
        assert cfg.temperature == 0.0

    def test_top_p_zero(self) -> None:
        cfg = GenerateConfig(top_p=0.0)
        assert cfg.top_p == 0.0

    def test_top_p_one(self) -> None:
        cfg = GenerateConfig(top_p=1.0)
        assert cfg.top_p == 1.0

    def test_min_p_zero(self) -> None:
        cfg = GenerateConfig(min_p=0.0)
        assert cfg.min_p == 0.0

    def test_min_p_one(self) -> None:
        cfg = GenerateConfig(min_p=1.0)
        assert cfg.min_p == 1.0

    def test_top_logprobs_max(self) -> None:
        cfg = GenerateConfig(top_logprobs=20)
        assert cfg.top_logprobs == 20

    def test_clear_cache_interval_zero_disabled(self) -> None:
        cfg = GenerateConfig(clear_cache_interval=0)
        assert cfg.clear_cache_interval == 0

    def test_prefill_step_size_one(self) -> None:
        cfg = GenerateConfig(prefill_step_size=1)
        assert cfg.prefill_step_size == 1


class TestGenerateConfigNegative:
    def test_negative_temperature(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(temperature=-0.1)

    def test_top_p_above_one(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(top_p=1.1)

    def test_top_p_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(top_p=-0.1)

    def test_min_p_above_one(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(min_p=1.1)

    def test_top_logprobs_above_20(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(top_logprobs=21)

    def test_top_logprobs_negative(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(top_logprobs=-1)

    def test_negative_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(max_tokens=-1)

    def test_prefill_step_size_zero(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(prefill_step_size=0)

    def test_repetition_penalty_below_one(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(repetition_penalty=0.5)


class TestGenerateConfigAlternateFlows:
    def test_compile_decode_on(self) -> None:
        cfg = GenerateConfig(compile_decode=True)
        assert cfg.compile_decode is True

    def test_warmup_after_load_on(self) -> None:
        cfg = GenerateConfig(warmup_after_load=True)
        assert cfg.warmup_after_load is True

    def test_stream_policy_overlap(self) -> None:
        cfg = GenerateConfig(stream_policy=StreamPolicy.OVERLAP)
        assert cfg.stream_policy == StreamPolicy.OVERLAP

    def test_stop_sequences_tuple(self) -> None:
        cfg = GenerateConfig(stop_sequences=("<|end|>", "STOP"))
        assert len(cfg.stop_sequences) == 2


# =============================================================================
# CacheConfig
# =============================================================================


class TestCacheConfigBoundary:
    def test_all_none_defaults(self) -> None:
        cfg = CacheConfig()
        assert cfg.max_kv_size is None
        assert cfg.kv_bits is None

    def test_kv_bits_set(self) -> None:
        cfg = CacheConfig(kv_bits=8)
        assert cfg.kv_bits == 8

    def test_quantized_kv_start_zero(self) -> None:
        cfg = CacheConfig(quantized_kv_start=0)
        assert cfg.quantized_kv_start == 0


class TestCacheConfigNegative:
    def test_quantized_kv_start_negative(self) -> None:
        with pytest.raises(ValidationError):
            CacheConfig(quantized_kv_start=-1)


# =============================================================================
# PromptCacheConfig
# =============================================================================


class TestPromptCacheConfigBoundary:
    def test_max_entries_zero(self) -> None:
        cfg = PromptCacheConfig(max_entries=0)
        assert cfg.max_entries == 0

    def test_target_rss_ratio_bounds(self) -> None:
        cfg = PromptCacheConfig(target_rss_ratio=0.01)
        assert cfg.target_rss_ratio == 0.01
        cfg2 = PromptCacheConfig(target_rss_ratio=1.0)
        assert cfg2.target_rss_ratio == 1.0

    def test_trim_step_one(self) -> None:
        cfg = PromptCacheConfig(trim_step=1)
        assert cfg.trim_step == 1


class TestPromptCacheConfigNegative:
    def test_target_rss_ratio_zero(self) -> None:
        with pytest.raises(ValidationError):
            PromptCacheConfig(target_rss_ratio=0.0)

    def test_target_rss_ratio_above_one(self) -> None:
        with pytest.raises(ValidationError):
            PromptCacheConfig(target_rss_ratio=1.1)

    def test_trim_step_zero(self) -> None:
        with pytest.raises(ValidationError):
            PromptCacheConfig(trim_step=0)


class TestPromptCacheConfigAlternateFlows:
    def test_memory_ceiling_policies(self) -> None:
        for policy in MemoryCeilingPolicy:
            cfg = PromptCacheConfig(on_memory_ceiling=policy)
            assert cfg.on_memory_ceiling == policy


# =============================================================================
# BatchConfig
# =============================================================================


class TestBatchConfigBoundary:
    def test_padding_side_left(self) -> None:
        cfg = BatchConfig(padding_side=PaddingSide.LEFT)
        assert cfg.padding_side == PaddingSide.LEFT

    def test_padding_side_right(self) -> None:
        cfg = BatchConfig(padding_side=PaddingSide.RIGHT)
        assert cfg.padding_side == PaddingSide.RIGHT

    def test_time_budget_zero(self) -> None:
        cfg = BatchConfig(time_budget_ms=0)
        assert cfg.time_budget_ms == 0

    def test_min_batch_size(self) -> None:
        cfg = BatchConfig(max_batch_size=1)
        assert cfg.max_batch_size == 1


class TestBatchConfigNegative:
    def test_prefill_batch_size_zero(self) -> None:
        with pytest.raises(ValidationError):
            BatchConfig(prefill_batch_size=0)

    def test_completion_batch_size_zero(self) -> None:
        with pytest.raises(ValidationError):
            BatchConfig(completion_batch_size=0)

    def test_max_batch_size_zero(self) -> None:
        with pytest.raises(ValidationError):
            BatchConfig(max_batch_size=0)


# =============================================================================
# SpeculativeConfig / ToolCallingConfig
# =============================================================================


class TestSpeculativeConfig:
    def test_defaults_disabled(self) -> None:
        cfg = SpeculativeConfig()
        assert cfg.draft_model_path is None
        assert cfg.num_draft_tokens == 5

    def test_custom_draft_tokens(self) -> None:
        cfg = SpeculativeConfig(num_draft_tokens=10)
        assert cfg.num_draft_tokens == 10

    def test_num_draft_tokens_min(self) -> None:
        cfg = SpeculativeConfig(num_draft_tokens=1)
        assert cfg.num_draft_tokens == 1

    def test_num_draft_tokens_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SpeculativeConfig(num_draft_tokens=0)


class TestToolCallingConfig:
    def test_default_parser(self) -> None:
        cfg = ToolCallingConfig()
        assert cfg.tool_call_parser == "generic"

    def test_custom_parser(self) -> None:
        cfg = ToolCallingConfig(tool_call_parser="qwen")
        assert cfg.tool_call_parser == "qwen"


# =============================================================================
# ServerConfig
# =============================================================================


class TestServerConfigBoundary:
    def test_port_min(self) -> None:
        cfg = ServerConfig(port=1)
        assert cfg.port == 1

    def test_port_max(self) -> None:
        cfg = ServerConfig(port=65535)
        assert cfg.port == 65535

    def test_queue_size_zero_unbounded(self) -> None:
        cfg = ServerConfig(max_queue_size=0)
        assert cfg.max_queue_size == 0


class TestServerConfigNegative:
    def test_port_zero(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig(port=0)

    def test_port_too_high(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig(port=65536)

    def test_workers_zero(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig(workers=0)


# =============================================================================
# AppConfig — cross-section / compatibility
# =============================================================================


class TestAppConfigCompatibility:
    def test_extra_fields_rejected(self) -> None:
        """Unknown fields fail (strict Pydantic model)."""
        with pytest.raises(ValidationError):
            AppConfig(unknown_section="value")

    def test_section_override_by_dict(self) -> None:
        cfg = AppConfig(model=ModelConfig(model_path="/tmp/model"))
        assert cfg.model.model_path == "/tmp/model"
        # Other sections remain default
        assert cfg.generate.max_tokens == 512

    def test_all_sections_present(self) -> None:
        cfg = AppConfig()
        assert cfg.model is not None
        assert cfg.generate is not None
        assert cfg.memory is not None
        assert cfg.cache is not None
        assert cfg.prompt_cache is not None
        assert cfg.batch is not None
        assert cfg.speculative is not None
        assert cfg.tool_calling is not None
        assert cfg.server is not None
        assert cfg.observability is not None

    def test_strict_validation_default_off(self) -> None:
        cfg = AppConfig()
        assert cfg.strict_validation is False
