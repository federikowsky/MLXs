"""Tests for error hierarchy (§6.9).

Patterns: happy path, boundary, alternate flows, compatibility.
"""

from __future__ import annotations

import pytest

from mlxs._errors import (
    CapacityExceededError,
    GenerationError,
    InvalidConfigError,
    InvalidPromptError,
    MemoryCeilingError,
    MLXsError,
    ModelLoadError,
    ModelNotLoadedError,
    RequestTimeoutError,
    StopSequenceError,
    TokenizerError,
)


class TestErrorHierarchy:
    """All errors inherit from MLXsError."""

    @pytest.mark.parametrize(
        "cls",
        [
            ModelLoadError,
            TokenizerError,
            ModelNotLoadedError,
            InvalidPromptError,
            InvalidConfigError,
            MemoryCeilingError,
            CapacityExceededError,
            RequestTimeoutError,
            GenerationError,
            StopSequenceError,
        ],
    )
    def test_inherits_from_base(self, cls: type) -> None:
        err = cls("test message")
        assert isinstance(err, MLXsError)
        assert isinstance(err, Exception)

    @pytest.mark.parametrize(
        "cls",
        [
            ModelLoadError,
            TokenizerError,
            ModelNotLoadedError,
            InvalidPromptError,
            InvalidConfigError,
            MemoryCeilingError,
            CapacityExceededError,
            RequestTimeoutError,
            GenerationError,
            StopSequenceError,
        ],
    )
    def test_message_preserved(self, cls: type) -> None:
        err = cls("specific error")
        assert err.message == "specific error"
        assert str(err) == "specific error"


class TestStatusHints:
    """Each error maps to correct HTTP status code for server."""

    def test_model_load_500(self) -> None:
        assert ModelLoadError.status_hint == 500

    def test_tokenizer_500(self) -> None:
        assert TokenizerError.status_hint == 500

    def test_model_not_loaded_503(self) -> None:
        assert ModelNotLoadedError.status_hint == 503

    def test_invalid_prompt_400(self) -> None:
        assert InvalidPromptError.status_hint == 400

    def test_invalid_config_400(self) -> None:
        assert InvalidConfigError.status_hint == 400

    def test_memory_ceiling_503(self) -> None:
        assert MemoryCeilingError.status_hint == 503

    def test_capacity_exceeded_503(self) -> None:
        assert CapacityExceededError.status_hint == 503

    def test_request_timeout_503(self) -> None:
        assert RequestTimeoutError.status_hint == 503

    def test_generation_500(self) -> None:
        assert GenerationError.status_hint == 500

    def test_stop_sequence_200(self) -> None:
        assert StopSequenceError.status_hint == 200


class TestErrorCatchability:
    """Errors can be caught at appropriate levels."""

    def test_catch_by_base(self) -> None:
        with pytest.raises(MLXsError):
            raise ModelLoadError("fail")

    def test_catch_by_specific(self) -> None:
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("fail")

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(CapacityExceededError, Exception)
