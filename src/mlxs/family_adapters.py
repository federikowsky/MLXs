from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from mlxs._types import ModelMode

_QWEN_VISION_PREFIXES = (
    "visual.",
    "vision_tower.",
    "vision_model.",
    "multi_modal_projector.",
    "mm_projector.",
)
_QWEN_VISION_EXACT = ("visual", "vision_tower", "vision_model")


@dataclass(frozen=True, slots=True)
class AdapterAliasMatch:
    source_names: tuple[str, ...]
    rule_id: str
    note: str | None = None


class FamilyAdapter(Protocol):
    adapter_name: str

    def alias_matches(
        self,
        target_name: str,
        *,
        model_mode: ModelMode,
    ) -> tuple[AdapterAliasMatch, ...]:
        ...


@dataclass(frozen=True, slots=True)
class QwenFamilyAdapter:
    adapter_name: str = "qwen_family"

    def alias_matches(
        self,
        target_name: str,
        *,
        model_mode: ModelMode,
    ) -> tuple[AdapterAliasMatch, ...]:
        del model_mode

        matches: list[AdapterAliasMatch] = []
        seen: set[tuple[str, ...]] = set()

        def add(source_names: tuple[str, ...], rule_id: str, note: str | None) -> None:
            normalized = tuple(name for name in source_names if name)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            matches.append(
                AdapterAliasMatch(
                    source_names=normalized,
                    rule_id=rule_id,
                    note=note,
                )
            )

        if not target_name.startswith(("language_model.", "vision_tower.")):
            add(
                (f"language_model.{target_name}",),
                "qwen_family:language_model_prefix",
                "Match pilot-family source tensors that retain the language_model prefix",
            )

        if target_name.startswith("vision_tower."):
            suffix = target_name[len("vision_tower.") :]
            add(
                (f"visual.{suffix}",),
                "qwen_family:visual_prefix",
                "Match Qwen-family public visual prefix",
            )
            add(
                (f"vision_model.{suffix}",),
                "qwen_family:vision_model_prefix",
                "Match Qwen-family legacy vision_model prefix",
            )

        return tuple(matches)


_QWEN_FAMILY_ADAPTER = QwenFamilyAdapter()
_FAMILY_ADAPTERS: dict[str, FamilyAdapter] = {
    "qwen2": _QWEN_FAMILY_ADAPTER,
    "qwen3": _QWEN_FAMILY_ADAPTER,
}


def get_family_adapter(model_type: str) -> FamilyAdapter | None:
    return _FAMILY_ADAPTERS.get(model_type)


def sanitize_qwen_family_weights(
    weights: Mapping[str, Any],
    *,
    model_mode: ModelMode,
    sanitize_text_weights: Callable[[dict[str, Any]], dict[str, Any]],
    vision_sanitize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if model_mode == ModelMode.TEXT:
        return sanitize_text_weights(_qwen_text_weights(weights))

    language_weights, vision_weights = _qwen_multimodal_partitions(weights)
    sanitized_language = sanitize_text_weights(language_weights)
    if vision_sanitize is not None:
        vision_weights = vision_sanitize(vision_weights)
    return sanitized_language | vision_weights


def _qwen_text_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    filtered = {
        key: value
        for key, value in weights.items()
        if key not in _QWEN_VISION_EXACT
        and not any(key.startswith(prefix) for prefix in _QWEN_VISION_PREFIXES)
    }
    return {
        key[len("language_model.") :] if key.startswith("language_model.") else key: value
        for key, value in filtered.items()
    }


def _qwen_multimodal_partitions(
    weights: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    language_weights: dict[str, Any] = {}
    vision_weights: dict[str, Any] = {}

    for key, value in weights.items():
        remapped = _remap_qwen_family_runtime_key(key)
        if remapped is None:
            continue
        bucket, remapped_key = remapped
        if bucket == "vision":
            vision_weights[remapped_key] = value
        else:
            language_weights[remapped_key] = value

    return language_weights, vision_weights


def _remap_qwen_family_runtime_key(key: str) -> tuple[str, str] | None:
    if key.startswith(("multi_modal_projector.", "mm_projector.")):
        return None

    if key.startswith("visual."):
        key = f"vision_tower.{key[len('visual.') :]}"
    elif key.startswith("vision_model."):
        key = f"vision_tower.{key[len('vision_model.') :]}"

    if key.startswith("vision_tower."):
        return ("vision", key)
    if key.startswith("language_model."):
        return ("language", key[len("language_model.") :])
    if key in _QWEN_VISION_EXACT:
        return None
    return ("language", key)
