"""Tests for MultimodalModelProtocol (FR12, §7.4)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlxs.protocols.multimodal import MultimodalModelProtocol


class _MockMultimodalModel:
    """Mock class that satisfies MultimodalModelProtocol."""

    def __call__(
        self, input_ids: mx.array, *, cache: Any = None, mask: Any = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        B, T = input_ids.shape
        return mx.zeros((B, T, 32))

    def prepare_inputs(
        self,
        input_ids: mx.array,
        *,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        pixel_attention_mask: mx.array | None = None,
        audio_features: mx.array | None = None,
        video_pixel_values: mx.array | None = None,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array | None]:
        if pixel_values is None:
            return input_ids, None
        B, T = input_ids.shape
        return input_ids, mx.zeros((B, T, 32))

    def make_cache(self) -> list[Any]:
        return []

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return 32

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights

    def parameters(self) -> dict[str, Any]:
        return {}

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def supports_audio(self) -> bool:
        return False

    @property
    def image_token_id(self) -> int | None:
        return 151655


class _TextOnlyModel:
    """Mock that does NOT satisfy MultimodalModelProtocol."""

    def __call__(self, input_ids: mx.array, *, cache: Any = None) -> mx.array:
        return mx.zeros((1, 4, 32))


def test_multimodal_protocol_isinstance() -> None:
    """A class with prepare_inputs satisfies the runtime check."""
    model = _MockMultimodalModel()
    assert isinstance(model, MultimodalModelProtocol)


def test_text_only_not_multimodal() -> None:
    """A text-only model without prepare_inputs fails the check."""
    model = _TextOnlyModel()
    assert not isinstance(model, MultimodalModelProtocol)


def test_prepare_inputs_no_media() -> None:
    """prepare_inputs with no pixel_values returns None embeddings."""
    model = _MockMultimodalModel()
    input_ids = mx.array([[1, 2, 3]])
    ids_out, embeds = model.prepare_inputs(input_ids)
    assert mx.array_equal(ids_out, input_ids)
    assert embeds is None


def test_prepare_inputs_with_media() -> None:
    """prepare_inputs with pixel_values returns embeddings."""
    model = _MockMultimodalModel()
    input_ids = mx.array([[1, 2, 3]])
    pixel_values = mx.zeros((1, 3, 14, 14))
    ids_out, embeds = model.prepare_inputs(input_ids, pixel_values=pixel_values)
    assert embeds is not None
    assert embeds.shape == (1, 3, 32)


def test_properties() -> None:
    """Properties return expected values."""
    model = _MockMultimodalModel()
    assert model.supports_vision is True
    assert model.supports_audio is False
    assert model.image_token_id == 151655
