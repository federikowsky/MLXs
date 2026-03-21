"""Shared multimodal helpers for Bucket 0 migration work.

These helpers are intentionally narrow:
- nested/flat multimodal config parsing,
- dual-mode vision construction gating,
- shared prepare_inputs skeleton with image/video branches.

They do not impose any family-specific projector, encoder, or sanitize logic.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Self, TypeVar

import mlx.core as mx

from mlxs._types import ModelMode
from mlxs.models.base import BaseModelArgs
from mlxs.models.vision import merge_embeddings

TextConfigT = TypeVar("TextConfigT")
LanguageModelT = TypeVar("LanguageModelT")
VisionTowerT = TypeVar("VisionTowerT")

TextConfigNormalizer = Callable[[dict[str, Any]], TextConfigT | dict[str, Any]]
LanguageBuilder = Callable[[], LanguageModelT]
VisionBuilder = Callable[[dict[str, Any]], VisionTowerT]
EmbeddingProvider = Callable[[mx.array], mx.array]
MediaEncoder = Callable[..., mx.array]


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _extract_optional_mapping(
    params: Mapping[str, Any],
    field_names: Sequence[str],
) -> dict[str, Any] | None:
    for field_name in field_names:
        if field_name not in params or params[field_name] is None:
            continue
        return _require_mapping(params[field_name], field_name=field_name)
    return None


@dataclass
class MultimodalArgsMixin(BaseModelArgs, Generic[TextConfigT]):
    """Reusable top-level args shape for wrapper-style multimodal families.

    Future family-specific args classes can inherit from this mixin and use
    ``from_flat_or_nested`` to preserve existing flat and nested config
    semantics without copying parser logic.
    """

    text_config: TextConfigT | dict[str, Any] | None = None
    vision_config: dict[str, Any] | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None

    @classmethod
    def from_flat_or_nested(
        cls,
        params: Mapping[str, Any],
        *,
        default_model_type: str,
        excluded_keys: Collection[str] = (),
        vision_config_keys: Sequence[str] = ("vision_config",),
        text_config_normalizer: TextConfigNormalizer[TextConfigT] | None = None,
    ) -> Self:
        """Parse raw config dicts with nested or flat text config semantics.

        Flat mode keeps recognized top-level multimodal keys out of the
        generated ``text_config`` payload. Nested mode preserves the provided
        ``text_config`` mapping as-is, optionally passing it through a caller
        normalizer for family-specific conversion.
        """

        constructor_keys = {
            name for name in inspect.signature(cls).parameters if name != "self"
        }
        reserved_keys = {"text_config", *vision_config_keys}
        top_level_keys = {
            key for key in constructor_keys if key in params and key not in reserved_keys
        }

        payload = {
            key: params[key]
            for key in top_level_keys
        }
        if "model_type" in constructor_keys:
            payload["model_type"] = params.get("model_type", default_model_type)

        raw_text_config = params.get("text_config")
        if raw_text_config is None:
            excluded = set(excluded_keys) | reserved_keys | top_level_keys
            text_config = {
                key: value
                for key, value in params.items()
                if key not in excluded
            }
        else:
            text_config = _require_mapping(raw_text_config, field_name="text_config")

        if text_config_normalizer is not None:
            payload["text_config"] = text_config_normalizer(text_config)
        else:
            payload["text_config"] = text_config

        vision_config = _extract_optional_mapping(params, vision_config_keys)
        if "vision_config" in constructor_keys:
            payload["vision_config"] = vision_config

        return cls(
            **{key: value for key, value in payload.items() if key in constructor_keys}
        )


@dataclass(frozen=True, slots=True)
class DualModeComponents(Generic[LanguageModelT, VisionTowerT]):
    """Result of minimal dual-mode construction gating."""

    language_model: LanguageModelT
    vision_tower: VisionTowerT | None
    model_mode: ModelMode

    @property
    def supports_vision(self) -> bool:
        return self.model_mode != ModelMode.TEXT and self.vision_tower is not None


def build_dual_mode_components(
    *,
    model_mode: ModelMode,
    language_builder: LanguageBuilder[LanguageModelT],
    vision_config: Mapping[str, Any] | None,
    vision_builder: VisionBuilder[VisionTowerT],
) -> DualModeComponents[LanguageModelT, VisionTowerT]:
    """Build the language side always and the vision side only when enabled."""

    language_model = language_builder()
    if model_mode == ModelMode.TEXT or vision_config is None:
        return DualModeComponents(
            language_model=language_model,
            vision_tower=None,
            model_mode=model_mode,
        )

    return DualModeComponents(
        language_model=language_model,
        vision_tower=vision_builder(dict(vision_config)),
        model_mode=model_mode,
    )


@dataclass(frozen=True, slots=True)
class MediaBranch:
    """One optional media branch for the shared prepare_inputs skeleton."""

    values: mx.array
    placeholder_token_id: int
    encode: MediaEncoder
    encoder_kwargs: Mapping[str, Any] = field(default_factory=dict)


def prepare_multimodal_inputs(
    input_ids: mx.array,
    *,
    embed_tokens: EmbeddingProvider,
    image_branch: MediaBranch | None = None,
    video_branch: MediaBranch | None = None,
) -> tuple[mx.array, mx.array | None]:
    """Shared image/video merge skeleton for multimodal wrappers.

    The order is stable and explicit: image branch first, then video branch.
    This preserves the existing `qwen3_5` image-then-video path without baking in any
    family-specific assumptions.
    """

    branches = tuple(
        branch for branch in (image_branch, video_branch) if branch is not None
    )
    if not branches:
        return input_ids, None

    merged_embeddings = embed_tokens(input_ids)
    for branch in branches:
        media_embeddings = branch.encode(branch.values, **branch.encoder_kwargs)
        merged_embeddings = merge_embeddings(
            merged_embeddings,
            media_embeddings,
            input_ids,
            branch.placeholder_token_id,
        )
    return input_ids, merged_embeddings
