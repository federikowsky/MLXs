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


@dataclass(frozen=True, slots=True)
class AdapterMappingMatch:
    source_names: tuple[str, ...]
    transforms: tuple[AdapterTransformSpec, ...]
    rule_id: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class AdapterTransformSpec:
    kind: str
    scalar: float | None = None
    axis: int | None = None
    permutation: tuple[int, ...] | None = None
    source_axis: int | None = None
    target_axis: int | None = None
    shape: tuple[int, ...] | None = None
    slice_start: int | None = None
    slice_stop: int | None = None
    dtype: str | None = None
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

    def mapping_matches(
        self,
        target_name: str,
        *,
        model_mode: ModelMode,
    ) -> tuple[AdapterMappingMatch, ...]:
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
        return _qwen_alias_matches(
            target_name,
            adapter_prefix="qwen_family",
            include_language_prefix=True,
        )

    def mapping_matches(
        self,
        target_name: str,
        *,
        model_mode: ModelMode,
    ) -> tuple[AdapterMappingMatch, ...]:
        del target_name, model_mode
        return ()


@dataclass(frozen=True, slots=True)
class Qwen35FamilyAdapter:
    adapter_name: str = "qwen35_family"

    def alias_matches(
        self,
        target_name: str,
        *,
        model_mode: ModelMode,
    ) -> tuple[AdapterAliasMatch, ...]:
        del model_mode
        return _qwen_alias_matches(
            target_name,
            adapter_prefix="qwen35_family",
            include_language_prefix=True,
        )

    def mapping_matches(
        self,
        target_name: str,
        *,
        model_mode: ModelMode,
    ) -> tuple[AdapterMappingMatch, ...]:
        del target_name, model_mode
        return ()


@dataclass(frozen=True, slots=True)
class Qwen35MoeFamilyAdapter:
    adapter_name: str = "qwen35_moe_family"

    def alias_matches(
        self,
        target_name: str,
        *,
        model_mode: ModelMode,
    ) -> tuple[AdapterAliasMatch, ...]:
        del model_mode

        matches = list(
            _qwen_alias_matches(
                target_name,
                adapter_prefix="qwen35_moe_family",
                include_language_prefix=False,
            )
        )
        if target_name.startswith("language_model."):
            stripped = target_name[len("language_model.") :]
            if stripped:
                matches.append(
                    AdapterAliasMatch(
                        source_names=(stripped,),
                        rule_id="qwen35_moe_family:strip_language_model_prefix",
                        note=(
                            "Match qwen3.5-MoE source tensors that omit the "
                            "language_model wrapper prefix"
                        ),
                    )
                )
        if target_name.startswith("language_model.model."):
            matches.append(
                AdapterAliasMatch(
                    source_names=(
                        target_name.replace(
                            "language_model.model.",
                            "model.language_model.",
                            1,
                        ),
                    ),
                    rule_id="qwen35_moe_family:model_language_model_prefix",
                    note=(
                        "Match qwen3.5-MoE source tensors normalized under "
                        "model.language_model.*"
                    ),
                )
            )
        return _dedupe_alias_matches(matches)

    def mapping_matches(
        self,
        target_name: str,
        *,
        model_mode: ModelMode,
    ) -> tuple[AdapterMappingMatch, ...]:
        del model_mode
        matches: list[AdapterMappingMatch] = []
        matches.extend(_qwen35_moe_gate_up_matches(target_name))
        matches.extend(_qwen35_moe_down_proj_matches(target_name))
        return tuple(matches)


_QWEN_FAMILY_ADAPTER = QwenFamilyAdapter()
_QWEN35_FAMILY_ADAPTER = Qwen35FamilyAdapter()
_QWEN35_MOE_FAMILY_ADAPTER = Qwen35MoeFamilyAdapter()
_FAMILY_ADAPTERS: dict[str, FamilyAdapter] = {
    "qwen2": _QWEN_FAMILY_ADAPTER,
    "qwen3": _QWEN_FAMILY_ADAPTER,
    "qwen3_5": _QWEN35_FAMILY_ADAPTER,
    "qwen3_5_moe": _QWEN35_MOE_FAMILY_ADAPTER,
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
        return sanitize_text_weights(
            _strip_language_model_prefix(_filter_qwen_text_weights(weights))
        )

    language_weights, vision_weights = _qwen_multimodal_partitions(weights)
    sanitized_language = sanitize_text_weights(
        _strip_language_model_prefix(language_weights)
    )
    if vision_sanitize is not None:
        vision_weights = vision_sanitize(vision_weights)
    return sanitized_language | vision_weights


def sanitize_qwen35_family_weights(
    weights: Mapping[str, Any],
    *,
    model_mode: ModelMode,
    tie_word_embeddings: bool,
    strip_language_model_prefix: bool,
    vision_sanitize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if model_mode == ModelMode.TEXT:
        language_weights = _filter_qwen_text_weights(weights)
        if strip_language_model_prefix:
            language_weights = _strip_language_model_prefix(language_weights)
        return sanitize_qwen35_text_weights(
            language_weights,
            tie_word_embeddings=tie_word_embeddings,
        )

    language_weights, vision_weights = _qwen_multimodal_partitions(weights)
    if strip_language_model_prefix:
        language_weights = _strip_language_model_prefix(language_weights)
    sanitized_language = sanitize_qwen35_text_weights(
        language_weights,
        tie_word_embeddings=tie_word_embeddings,
    )
    if vision_sanitize is not None:
        vision_weights = vision_sanitize(vision_weights)
    return sanitized_language | vision_weights


def sanitize_qwen35_moe_family_weights(
    weights: Mapping[str, Any],
    *,
    model_mode: ModelMode,
    tie_word_embeddings: bool,
    vision_sanitize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if model_mode == ModelMode.TEXT:
        return sanitize_qwen35_moe_language_weights(
            _filter_qwen_text_weights(weights),
            tie_word_embeddings=tie_word_embeddings,
        )

    language_weights, vision_weights = _qwen_multimodal_partitions(weights)
    sanitized_language = sanitize_qwen35_moe_language_weights(
        language_weights,
        tie_word_embeddings=tie_word_embeddings,
    )
    if vision_sanitize is not None:
        vision_weights = vision_sanitize(vision_weights)
    return sanitized_language | vision_weights


def sanitize_qwen35_text_weights(
    weights: dict[str, Any],
    *,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    has_mtp_weights = any("mtp." in key for key in weights)
    has_unsanitized_conv1d = any(
        "conv1d.weight" in key and value.shape[-1] != 1
        for key, value in weights.items()
    )
    should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
    sanitized = {key: value for key, value in weights.items() if "mtp." not in key}
    if tie_word_embeddings:
        sanitized.pop("lm_head.weight", None)
        sanitized.pop("language_model.lm_head.weight", None)

    norm_keys = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for key, value in list(sanitized.items()):
        if "conv1d.weight" in key and value.shape[-1] != 1:
            sanitized[key] = value.moveaxis(2, 1)
        if (
            should_shift_norm_weights
            and any(key.endswith(suffix) for suffix in norm_keys)
            and value.ndim == 1
        ):
            sanitized[key] = value + 1.0
    return sanitized


def sanitize_qwen35_moe_language_weights(
    weights: Mapping[str, Any],
    *,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    normalized = _normalize_qwen35_moe_language_weights(weights)
    split = _split_qwen35_moe_expert_weights(normalized)
    return sanitize_qwen35_text_weights(
        split,
        tie_word_embeddings=tie_word_embeddings,
    )


def _qwen_alias_matches(
    target_name: str,
    *,
    adapter_prefix: str,
    include_language_prefix: bool,
) -> tuple[AdapterAliasMatch, ...]:
    matches: list[AdapterAliasMatch] = []
    if include_language_prefix and not target_name.startswith(
        ("language_model.", "vision_tower.")
    ):
        matches.append(
            AdapterAliasMatch(
                source_names=(f"language_model.{target_name}",),
                rule_id=f"{adapter_prefix}:language_model_prefix",
                note=(
                    "Match family source tensors that retain the "
                    "language_model prefix"
                ),
            )
        )

    if target_name.startswith("vision_tower."):
        suffix = target_name[len("vision_tower.") :]
        matches.extend(
            (
                AdapterAliasMatch(
                    source_names=(f"visual.{suffix}",),
                    rule_id=f"{adapter_prefix}:visual_prefix",
                    note="Match family public visual prefix",
                ),
                AdapterAliasMatch(
                    source_names=(f"vision_model.{suffix}",),
                    rule_id=f"{adapter_prefix}:vision_model_prefix",
                    note="Match family legacy vision_model prefix",
                ),
            )
        )

    return _dedupe_alias_matches(matches)


def _dedupe_alias_matches(
    matches: list[AdapterAliasMatch],
) -> tuple[AdapterAliasMatch, ...]:
    seen: set[tuple[str, ...]] = set()
    ordered: list[AdapterAliasMatch] = []
    for match in matches:
        normalized = tuple(name for name in match.source_names if name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(
            AdapterAliasMatch(
                source_names=normalized,
                rule_id=match.rule_id,
                note=match.note,
            )
        )
    return tuple(ordered)


def _filter_qwen_text_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in weights.items()
        if key not in _QWEN_VISION_EXACT
        and not any(key.startswith(prefix) for prefix in _QWEN_VISION_PREFIXES)
    }


def _strip_language_model_prefix(weights: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key[len("language_model.") :] if key.startswith("language_model.") else key: value
        for key, value in weights.items()
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


def _normalize_qwen35_moe_language_weights(
    weights: Mapping[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in weights.items():
        if key.startswith("model.language_model"):
            key = key.replace("model.language_model", "language_model.model", 1)
        elif not key.startswith("language_model."):
            key = "language_model." + key
        normalized[key] = value
    return normalized


def _split_qwen35_moe_expert_weights(
    weights: Mapping[str, Any],
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in weights.items():
        gate_prefix = _strip_suffix(
            key,
            ".experts.gate_up_proj.weight",
        ) or _strip_suffix(key, ".experts.gate_up_proj")
        if gate_prefix is not None:
            mid = value.shape[-2] // 2
            sanitized[f"{gate_prefix}.switch_mlp.gate_proj.weight"] = value[..., :mid, :]
            sanitized[f"{gate_prefix}.switch_mlp.up_proj.weight"] = value[..., mid:, :]
            continue

        down_prefix = _strip_suffix(
            key,
            ".experts.down_proj.weight",
        ) or _strip_suffix(key, ".experts.down_proj")
        if down_prefix is not None:
            sanitized[f"{down_prefix}.switch_mlp.down_proj.weight"] = value
            continue

        sanitized[key] = value
    return sanitized


def _qwen35_moe_gate_up_matches(target_name: str) -> tuple[AdapterMappingMatch, ...]:
    specs = (
        (
            ".switch_mlp.gate_proj.weight",
            ".experts.gate_up_proj.weight",
            ".experts.gate_up_proj",
            True,
            "qwen35_moe_family:gate_up_split",
        ),
        (
            ".switch_mlp.up_proj.weight",
            ".experts.gate_up_proj.weight",
            ".experts.gate_up_proj",
            False,
            "qwen35_moe_family:gate_up_split",
        ),
    )
    matches: list[AdapterMappingMatch] = []
    for target_suffix, source_suffix, source_suffix_legacy, is_gate, rule_id in specs:
        prefix = _strip_suffix(target_name, target_suffix)
        if prefix is None:
            continue
        for candidate in _qwen35_moe_source_candidates(
            prefix,
            source_suffix,
            source_suffix_legacy,
        ):
            matches.append(
                AdapterMappingMatch(
                    source_names=(candidate,),
                    transforms=(
                        AdapterTransformSpec(
                            kind="slice",
                            axis=-2,
                            slice_start=0 if is_gate else None,
                            slice_stop=None,
                            note=(
                                "Split qwen3.5-MoE combined expert gate_up tensor "
                                "first half"
                                if is_gate
                                else "Split qwen3.5-MoE combined expert gate_up tensor "
                                "second half"
                            ),
                        ),
                    ),
                    rule_id=rule_id,
                    note="qwen35_moe_gate_up_split",
                )
            )
    return tuple(matches)


def _qwen35_moe_down_proj_matches(target_name: str) -> tuple[AdapterMappingMatch, ...]:
    prefix = _strip_suffix(target_name, ".switch_mlp.down_proj.weight")
    if prefix is None:
        return ()
    return tuple(
        AdapterMappingMatch(
            source_names=(candidate,),
            transforms=(),
            rule_id="qwen35_moe_family:down_proj_alias",
            note="qwen35_moe_down_proj_alias",
        )
        for candidate in _qwen35_moe_source_candidates(
            prefix,
            ".experts.down_proj.weight",
            ".experts.down_proj",
        )
    )


def _qwen35_moe_source_candidates(
    prefix: str,
    primary_suffix: str,
    legacy_suffix: str,
) -> tuple[str, ...]:
    candidates = [f"{prefix}{primary_suffix}", f"{prefix}{legacy_suffix}"]
    if prefix.startswith("language_model."):
        stripped = prefix[len("language_model.") :]
        candidates.extend((f"{stripped}{primary_suffix}", f"{stripped}{legacy_suffix}"))
    ordered: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return tuple(ordered)


def _strip_suffix(value: str, suffix: str) -> str | None:
    if not value.endswith(suffix):
        return None
    return value[: -len(suffix)]
