"""Multimodal model protocol — contract for VL/audio models (FR12, §7.4).

Extends ModelProtocol with ``prepare_inputs`` for media encoding and
embedding merge. Only VL model wrappers implement this protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx

from mlxs.protocols.model import ModelProtocol


@runtime_checkable
class MultimodalModelProtocol(ModelProtocol, Protocol):
    """Contract for models that support media inputs (vision, audio, video).

    Models implementing this protocol can process images/audio/video
    alongside text via ``prepare_inputs``, which produces merged
    ``input_embeddings`` for the forward pass.
    """

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
        """Encode media and merge with text embeddings.

        Args:
            input_ids: Token ids ``(B, T)`` with media placeholder tokens.
            pixel_values: Preprocessed images for vision models.
            image_grid_thw: Grid dimensions ``(N, 3)`` for Qwen-style models.
            pixel_attention_mask: Attention mask for variable-resolution images.
            audio_features: Preprocessed audio features.
            video_pixel_values: Preprocessed video frames.

        Returns:
            Tuple of ``(input_ids, input_embeddings)``. When no media is
            present, ``input_embeddings`` is ``None``.
        """
        ...

    @property
    def supports_vision(self) -> bool:
        """Whether the model has a loaded vision encoder."""
        ...

    @property
    def supports_audio(self) -> bool:
        """Whether the model has a loaded audio encoder."""
        ...

    @property
    def image_token_id(self) -> int | None:
        """Placeholder token id for images, or None if vision not supported."""
        ...
