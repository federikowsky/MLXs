"""Tests for config schema — defaults, validation, and immutability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlxs._types import (
    MemoryCeilingPolicy,
    ModelMode,
    PaddingSide,
    StreamPolicy,
    WeightFormat,
)
from mlxs.config.schema import (
    AppConfig,
    BatchConfig,
    CacheConfig,
    GenerateConfig,
    ModelConfig,
    ObservabilityConfig,
    PromptCacheConfig,
    ServerConfig,
)


class TestAppConfigDefaults:
    """All default values match specs.md §8.2."""

    def test_creates_with_all_defaults(self) -> None:
        config = AppConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.generate, GenerateConfig)
        assert isinstance(config.cache, CacheConfig)
        assert config.strict_validation is False

    def test_model_defaults(self) -> None:
        m = ModelConfig()
        assert m.model_path == ""
        assert m.adapter_path is None
        assert m.trust_remote_code is False
        assert m.lazy_load is True
        assert m.preload is False
        assert m.model_mode == ModelMode.AUTO
        assert m.weight_format == WeightFormat.AUTO
        assert m.model_hf_revision is None
        assert m.model_hf_token is None

    def test_generate_defaults(self) -> None:
        g = GenerateConfig()
        assert g.max_tokens == 512
        assert g.temperature == 1.0
        assert g.top_p == 1.0
        assert g.top_k == 0
        assert g.min_p == 0.0
        assert g.seed is None
        assert g.prefill_step_size == 2048
        assert g.clear_cache_interval == 256
        assert g.compile_decode is False
        assert g.warmup_after_load is False
        assert g.stream_policy == StreamPolicy.SINGLE
        assert g.repetition_penalty == 1.0
        assert g.stop_sequences == ()
        assert g.extra_eos_token_ids == ()

    def test_cache_defaults(self) -> None:
        c = CacheConfig()
        assert c.max_kv_size is None
        assert c.kv_bits is None
        assert c.kv_group_size == 64
        assert c.quantized_kv_start == 0

    def test_prompt_cache_defaults(self) -> None:
        pc = PromptCacheConfig()
        assert pc.enabled is True
        assert pc.max_entries == 100
        assert pc.on_memory_ceiling == MemoryCeilingPolicy.TRIM_CACHE
        assert pc.trim_step == 1
        assert pc.target_rss_ratio == 0.9

    def test_batch_defaults(self) -> None:
        b = BatchConfig()
        assert b.padding_side == PaddingSide.LEFT
        assert b.completion_batch_size == 4
        assert b.max_batch_size == 8

    def test_server_defaults(self) -> None:
        s = ServerConfig()
        assert s.host == "127.0.0.1"
        assert s.port == 8080
        assert s.max_concurrent_requests == 2
        assert s.request_timeout == 300.0

    def test_observability_defaults(self) -> None:
        o = ObservabilityConfig()
        assert o.log_level == "INFO"
        assert o.metrics_enabled is False


class TestConfigImmutability:
    """Config objects are frozen after construction (§8.1)."""

    def test_app_config_frozen(self) -> None:
        config = AppConfig()
        with pytest.raises(ValidationError):
            config.strict_validation = True  # type: ignore[misc]

    def test_model_config_frozen(self) -> None:
        m = ModelConfig()
        with pytest.raises(ValidationError):
            m.model_path = "/new/path"  # type: ignore[misc]

    def test_generate_config_frozen(self) -> None:
        g = GenerateConfig()
        with pytest.raises(ValidationError):
            g.temperature = 0.5  # type: ignore[misc]


class TestConfigValidation:
    """Pydantic validation catches invalid values."""

    def test_temperature_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(temperature=-1.0)

    def test_top_p_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(top_p=1.5)

    def test_port_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            ServerConfig(port=99999)

    def test_max_tokens_negative(self) -> None:
        with pytest.raises(ValidationError):
            GenerateConfig(max_tokens=-1)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(nonexistent_field="bad")  # type: ignore[call-arg]

    def test_target_rss_ratio_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PromptCacheConfig(target_rss_ratio=0.0)
        with pytest.raises(ValidationError):
            PromptCacheConfig(target_rss_ratio=1.5)


class TestConfigOverrides:
    """Config sections can be constructed with custom values."""

    def test_model_with_path(self) -> None:
        m = ModelConfig(model_path="/tmp/model")
        assert m.model_path == "/tmp/model"

    def test_app_config_with_section_override(self) -> None:
        config = AppConfig(model=ModelConfig(model_path="/tmp/model"))
        assert config.model.model_path == "/tmp/model"
        # Other sections remain defaults
        assert config.generate.temperature == 1.0

    def test_app_config_from_dict(self) -> None:
        config = AppConfig(**{"model": {"model_path": "/tmp/m"}, "strict_validation": True})
        assert config.model.model_path == "/tmp/m"
        assert config.strict_validation is True
