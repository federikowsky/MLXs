from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

from mlxs.convert.errors import MissingRequiredConfigError
from mlxs.convert.types import (
    CanonicalIR,
    ConversionOptions,
    ConversionPhase,
    DensityKind,
    InspectionReport,
    IRAmbiguities,
    IRConfig,
    IRConversion,
    IREvidence,
    IRIdentity,
    IRSource,
    IRTensorLayout,
    IRTokenizer,
    IRTopology,
    MacroTemplate,
    Modality,
)

_SUPPORTED_RUNTIME_TEMPLATES = {
    MacroTemplate.DECODER_DENSE,
    MacroTemplate.DECODER_MOE,
    MacroTemplate.MULTIMODAL_DECODER,
    MacroTemplate.SSM_HYBRID,
}

_REQUIRED_FIELDS_BY_TEMPLATE: dict[MacroTemplate, tuple[str, ...]] = {
    MacroTemplate.DECODER_DENSE: (
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
    ),
    MacroTemplate.DECODER_MOE: (
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "num_experts",
    ),
    MacroTemplate.MULTIMODAL_DECODER: (
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
    ),
    MacroTemplate.SSM_HYBRID: (
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
    ),
    MacroTemplate.ENCODER_DECODER: ("hidden_size",),
    MacroTemplate.MULTIMODAL_ENCODER_DECODER: ("hidden_size",),
}

_CONVERGED_RUNTIME_TARGETS: dict[str, str] = {
    "lfm2": "lfm2",
    "lfm2_vl": "lfm2",
    "qwen2": "qwen2",
    "qwen2_vl": "qwen2",
    "qwen3": "qwen3",
    "qwen3_5": "qwen3_5",
    "qwen3_5_vl": "qwen3_5",
    "qwen3_moe": "qwen3_moe",
    "qwen3_vl": "qwen3",
    "qwen3_vl_moe": "qwen3_moe",
}

_DEFERRED_MULTIMODAL_BOUNDARY_TARGETS = {"qwen3_5_moe"}


def normalize_inspection(
    inspection: InspectionReport,
    *,
    options: ConversionOptions | None = None,
) -> CanonicalIR:
    opts = options or ConversionOptions()
    config = inspection.config
    matched_config_keys = tuple(sorted(config.keys()))
    runtime_target_model_type = _resolve_runtime_target_model_type(config)
    macro_template, ambiguity_flags = _select_macro_template(inspection, runtime_target_model_type)
    topology = _build_topology(inspection, config, macro_template)
    canonical_values = _canonical_config_values(config)
    required_fields = _REQUIRED_FIELDS_BY_TEMPLATE[macro_template]
    missing = tuple(field for field in required_fields if canonical_values.get(field) is None)
    if missing:
        raise MissingRequiredConfigError(
            f"Missing required canonical config fields: {', '.join(missing)}",
            phase=ConversionPhase.NORMALIZATION,
            details={"missing_fields": missing},
        )

    source_profiles, aliases, tensor_groups = _detect_tensor_profiles(inspection.tensor_infos)
    tokenizer_type = _detect_tokenizer_type(inspection.tokenizer_artifacts)
    tokenizer_specials = _collect_special_tokens(config)
    is_supported = (
        runtime_target_model_type is not None
        and macro_template in _SUPPORTED_RUNTIME_TEMPLATES
        and _runtime_supports_modality(runtime_target_model_type, topology.multimodal)
    )
    architecture_label = runtime_target_model_type or config.get("model_type", "unknown")

    return CanonicalIR(
        source=IRSource(
            kind=inspection.source_kind,
            source_id=inspection.source_id,
            resolved_path=inspection.resolved_path,
            weight_format=inspection.weight_format,
            sharded=inspection.sharded,
            shard_list=inspection.shard_files,
            source_config_artifacts_present=("config.json",),
            tokenizer_artifacts_present=inspection.tokenizer_artifacts,
            multimodal_artifacts_present=inspection.multimodal_artifacts,
            custom_code_indicators=inspection.custom_code_indicators,
        ),
        identity=IRIdentity(
            macro_template=macro_template,
            modality=Modality.MULTIMODAL if topology.multimodal else Modality.TEXT,
            architecture_label=architecture_label,
            variant_label=(
                config.get("model_type")
                if config.get("model_type") != architecture_label
                else None
            ),
            runtime_target_model_type=architecture_label,
            supported_by_runtime=is_supported,
        ),
        topology=topology,
        config=IRConfig(
            values=canonical_values,
            raw_config=config,
            required_fields=required_fields,
        ),
        tensor_layout=IRTensorLayout(
            selected_source_tensor_profile=(
                source_profiles[0] if source_profiles else "runtime_native"
            ),
            canonical_internal_tensor_profile="runtime_post_sanitize",
            tensor_groups_present=tensor_groups,
            optional_tensor_groups_present=tuple(
                sorted(group for group in tensor_groups if group not in {"core", "lm_head"})
            ),
            missing_required_groups=(),
            fused_split_markers=tuple(sorted(source_profiles)),
            naming_aliases_discovered=aliases,
        ),
        tokenizer=IRTokenizer(
            tokenizer_type=tokenizer_type,
            artifact_paths=inspection.tokenizer_artifacts,
            special_tokens=tokenizer_specials,
            vocab_metadata={"vocab_size": canonical_values.get("vocab_size")},
            chat_template_present=any(
                "template" in path for path in inspection.tokenizer_artifacts
            ),
        ),
        conversion=IRConversion(
            dtype_policy=opts.dtype_policy,
            target_format_version=opts.target_format_version,
            quantization_mode=opts.quantization_mode,
            shard_output_policy="single" if opts.max_shard_bytes is None else "sharded",
            cast_rules=(opts.dtype_policy,),
            save_options={"max_shard_bytes": opts.max_shard_bytes},
            canonical_output_config=_canonicalize_output_config(config, architecture_label),
        ),
        evidence=IREvidence(
            matched_config_keys=matched_config_keys,
            matched_tensor_patterns=source_profiles,
            matched_shape_traits=tuple(_shape_traits(inspection.tensor_infos)),
            matched_tokenizer_traits=tuple(sorted(inspection.tokenizer_artifacts)),
            selected_rules=source_profiles,
            model_assisted_normalization_usage=(
                None if opts.model_assistance.value == "disabled" else opts.model_assistance.value
            ),
        ),
        ambiguities=IRAmbiguities(
            ambiguity_flags=ambiguity_flags,
            selected_resolution_path="deterministic_rules",
            confidence_score=None,
            fallback_markers=(),
        ),
    )


def _resolve_runtime_target_model_type(config: dict[str, Any]) -> str | None:
    from mlxs.load.registry import _MODEL_REMAPPING, MODEL_REGISTRY

    source_model_type = config.get("model_type")
    if not isinstance(source_model_type, str):
        return None
    canonical = _MODEL_REMAPPING.get(source_model_type, source_model_type)
    if canonical in _CONVERGED_RUNTIME_TARGETS:
        return _CONVERGED_RUNTIME_TARGETS[canonical]
    if _has_multimodal_config(config):
        if canonical in _DEFERRED_MULTIMODAL_BOUNDARY_TARGETS:
            return canonical
        multimodal_candidates: list[str] = []
        if canonical.endswith("_moe"):
            multimodal_candidates.append(canonical.replace("_moe", "_vl_moe"))
        multimodal_candidates.append(f"{canonical}_vl")
        for candidate in multimodal_candidates:
            if candidate in MODEL_REGISTRY:
                return candidate
    if canonical in MODEL_REGISTRY:
        return canonical
    return source_model_type


def _runtime_supports_modality(runtime_target_model_type: str, multimodal: bool) -> bool:
    if not multimodal:
        return True

    from mlxs.load.registry import get_model_classes

    try:
        ModelClass, _ = get_model_classes(runtime_target_model_type)
    except ValueError:
        return False
    signature = inspect.signature(ModelClass.__init__)
    return "model_mode" in signature.parameters


def _has_multimodal_config(config: dict[str, Any]) -> bool:
    multimodal_keys = {
        "vision_config",
        "visual_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    }
    return any(key in config for key in multimodal_keys)


def _select_macro_template(
    inspection: InspectionReport,
    runtime_target_model_type: str | None,
) -> tuple[MacroTemplate, tuple[str, ...]]:
    config = inspection.config
    source_model_type = config.get("model_type", "")
    tensor_names = {tensor.name for tensor in inspection.tensor_infos}
    ambiguity_flags: list[str] = []

    is_multimodal = _is_multimodal(config, inspection.multimodal_artifacts, tensor_names)
    is_encoder_decoder = _is_encoder_decoder(config, tensor_names)
    is_ssm_hybrid = _is_ssm_hybrid(config, tensor_names, runtime_target_model_type)
    is_moe = _is_moe(config, tensor_names)

    matches = [
        ("multimodal", is_multimodal),
        ("encoder_decoder", is_encoder_decoder),
        ("ssm_hybrid", is_ssm_hybrid),
        ("moe", is_moe),
    ]
    if sum(1 for _, matched in matches if matched) > 1:
        ambiguity_flags.append("multi_profile_match")

    if is_multimodal and is_encoder_decoder:
        return MacroTemplate.MULTIMODAL_ENCODER_DECODER, tuple(ambiguity_flags)
    if is_encoder_decoder:
        return MacroTemplate.ENCODER_DECODER, tuple(ambiguity_flags)
    if is_multimodal:
        return MacroTemplate.MULTIMODAL_DECODER, tuple(ambiguity_flags)
    if is_ssm_hybrid:
        return MacroTemplate.SSM_HYBRID, tuple(ambiguity_flags)
    if is_moe:
        return MacroTemplate.DECODER_MOE, tuple(ambiguity_flags)
    if source_model_type:
        return MacroTemplate.DECODER_DENSE, tuple(ambiguity_flags)
    ambiguity_flags.append("missing_model_type")
    return MacroTemplate.DECODER_DENSE, tuple(ambiguity_flags)


def _build_topology(
    inspection: InspectionReport,
    config: dict[str, Any],
    macro_template: MacroTemplate,
) -> IRTopology:
    attention_heads = _value_from_nested(config, "num_attention_heads")
    kv_heads = _value_from_nested(config, "num_key_value_heads", "num_kv_heads", "n_kv_heads")
    if attention_heads is not None and kv_heads is not None:
        if kv_heads == 1:
            attention_variant = "mqa"
        elif kv_heads < attention_heads:
            attention_variant = "gqa"
        else:
            attention_variant = "mha"
    else:
        attention_variant = None

    density = DensityKind.DENSE
    if macro_template == MacroTemplate.DECODER_MOE:
        density = DensityKind.MOE
    elif macro_template == MacroTemplate.SSM_HYBRID:
        density = DensityKind.HYBRID

    tensor_names = {tensor.name for tensor in inspection.tensor_infos}
    qkv_layout = "fused_qkv" if any("qkv" in name for name in tensor_names) else "split_qkv"
    recurrent_traits = tuple(
        sorted(
            marker
            for marker, present in {
                "arrays_cache": any(
                    "conv1d.weight" in name or "conv_1d.weight" in name
                    for name in tensor_names
                ),
                "ssm_state": any("A_log" in name or "rg_lru" in name for name in tensor_names),
                "hybrid_attention": any("self_attn" in name for name in tensor_names),
            }.items()
            if present
        )
    )
    return IRTopology(
        backbone_type=_backbone_type(macro_template),
        decoder_only=macro_template not in {
            MacroTemplate.ENCODER_DECODER,
            MacroTemplate.MULTIMODAL_ENCODER_DECODER,
        },
        encoder_decoder=macro_template in {
            MacroTemplate.ENCODER_DECODER,
            MacroTemplate.MULTIMODAL_ENCODER_DECODER,
        },
        multimodal=macro_template in {
            MacroTemplate.MULTIMODAL_DECODER,
            MacroTemplate.MULTIMODAL_ENCODER_DECODER,
        },
        ssm_hybrid=macro_template == MacroTemplate.SSM_HYBRID,
        density=density,
        attention_presence=macro_template != MacroTemplate.SSM_HYBRID or any(
            "self_attn" in name for name in tensor_names
        ),
        attention_variant=attention_variant,
        qkv_layout=qkv_layout,
        mlp_variant=_detect_mlp_variant(config, tensor_names, density),
        norm_type=_detect_norm_type(config),
        rope_variant=_detect_rope_variant(config, tensor_names),
        embedding_tying=_value_from_nested(config, "tie_word_embeddings"),
        projector_presence=any(
            "projector" in name or "projection" in name for name in inspection.multimodal_artifacts
        ) or any("multi_modal_projector" in name for name in tensor_names),
        tower_presence=any(
            "vision" in name or "audio" in name for name in inspection.multimodal_artifacts
        ) or any("vision_tower" in name or "visual." in name for name in tensor_names),
        recurrent_traits=recurrent_traits,
    )


def _is_multimodal(
    config: dict[str, Any],
    multimodal_artifacts: tuple[str, ...],
    tensor_names: set[str],
) -> bool:
    return bool(
        any(key in config for key in ("vision_config", "visual_config", "audio_config"))
        or multimodal_artifacts
        or any(
            name.startswith(prefix)
            for prefix in ("vision_tower.", "visual.", "multi_modal_projector.", "model.vision_")
            for name in tensor_names
        )
    )


def _is_encoder_decoder(config: dict[str, Any], tensor_names: set[str]) -> bool:
    encoder_decoder_keys = {
        "is_encoder_decoder",
        "encoder_layers",
        "decoder_layers",
        "decoder_start_token_id",
    }
    if any(key in config for key in encoder_decoder_keys):
        return True
    return any(
        name.startswith("encoder.") or name.startswith("decoder.")
        for name in tensor_names
    )


def _is_ssm_hybrid(
    config: dict[str, Any],
    tensor_names: set[str],
    runtime_target_model_type: str | None,
) -> bool:
    config_markers = (
        "state_size",
        "d_state",
        "mamba_d_state",
        "attention_window_size",
        "linear_attn_config",
        "layers_block_type",
        "block_types",
    )
    if any(marker in config for marker in config_markers):
        return True
    if runtime_target_model_type in {
        "mamba",
        "mamba2",
        "recurrent_gemma",
        "rwkv7",
        "jamba",
        "granitemoehybrid",
        "kimi_linear",
        "qwen3_5_moe",
    }:
        return True
    return any(
        "conv1d.weight" in name
        or "conv_1d.weight" in name
        or "A_log" in name
        or "rg_lru" in name
        for name in tensor_names
    )


def _is_moe(config: dict[str, Any], tensor_names: set[str]) -> bool:
    num_experts = _value_from_nested(
        config,
        "num_experts",
        "num_local_experts",
    )
    if isinstance(num_experts, int) and num_experts > 0:
        return True
    return any(
        ".experts." in name
        or "block_sparse_moe" in name
        or "switch_mlp" in name
        for name in tensor_names
    )


def _canonical_config_values(config: dict[str, Any]) -> dict[str, Any]:
    values = {
        "hidden_size": _value_from_nested(config, "hidden_size"),
        "num_hidden_layers": _value_from_nested(config, "num_hidden_layers"),
        "num_attention_heads": _value_from_nested(config, "num_attention_heads"),
        "num_key_value_heads": _value_from_nested(
            config, "num_key_value_heads", "num_kv_heads", "n_kv_heads"
        ),
        "intermediate_size": _value_from_nested(config, "intermediate_size"),
        "vocab_size": _value_from_nested(config, "vocab_size"),
        "max_position_embeddings": _value_from_nested(
            config, "max_position_embeddings", "model_max_length"
        ),
        "norm_epsilon": _value_from_nested(
            config, "rms_norm_eps", "layer_norm_epsilon", "norm_epsilon"
        ),
        "rope_theta": _value_from_nested(config, "rope_theta"),
        "rope_scaling": _value_from_nested(config, "rope_scaling", "rope_parameters"),
        "num_experts": _value_from_nested(config, "num_local_experts", "num_experts"),
        "num_experts_per_tok": _value_from_nested(
            config, "num_experts_per_tok", "num_experts_per_token"
        ),
        "tie_word_embeddings": _value_from_nested(config, "tie_word_embeddings"),
        "vision_hidden_size": _value_from_nested(
            config, "hidden_size", nested_key="vision_config"
        ),
        "projector_hidden_size": _value_from_nested(
            config, "hidden_size", nested_key="vision_config"
        ),
    }
    return {key: value for key, value in values.items() if value is not None}


def _canonicalize_output_config(
    config: dict[str, Any],
    runtime_target_model_type: str,
) -> dict[str, Any]:
    output = dict(config)
    output["model_type"] = runtime_target_model_type
    if isinstance(output.get("text_config"), dict):
        output["text_config"] = dict(output["text_config"])
    if isinstance(output.get("vision_config"), dict):
        output["vision_config"] = dict(output["vision_config"])
    if isinstance(output.get("visual_config"), dict):
        output["visual_config"] = dict(output["visual_config"])
    return output


def _detect_tensor_profiles(
    tensor_infos: Iterable[Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    names = {tensor.name for tensor in tensor_infos}
    profiles: list[str] = []
    aliases: list[str] = []
    groups = {"core"}
    if any(name.startswith("language_model.") for name in names):
        profiles.append("language_model_prefixed")
        aliases.append("language_model")
    if any(name.startswith("visual.") for name in names):
        profiles.append("visual_prefix")
        aliases.append("visual->vision_tower")
    if any("multi_modal_projector" in name or "vision_projection" in name for name in names):
        profiles.append("projector_alias")
        groups.add("multimodal")
    if any(".experts." in name and ".w1." in name for name in names):
        profiles.append("moe_triplets")
        groups.add("moe")
    if any("gate_up_proj" in name for name in names):
        profiles.append("gate_up_split")
    if any("shared_mlp.input_linear.weight" in name for name in names):
        profiles.append("shared_mlp_split")
    if any("conv1d.weight" in name or "conv_1d.weight" in name for name in names):
        profiles.append("conv_axis_sensitive")
        groups.add("ssm")
    if any(name == "lm_head.weight" for name in names):
        groups.add("lm_head")
    if any("rotary_emb.inv_freq" in name for name in names):
        profiles.append("rotary_inv_freq")
    if not profiles:
        profiles.append("runtime_native")
    return tuple(profiles), tuple(sorted(set(aliases))), tuple(sorted(groups))


def _detect_tokenizer_type(tokenizer_artifacts: tuple[str, ...]) -> str:
    if "tokenizer.model" in tokenizer_artifacts:
        return "sentencepiece"
    if "tokenizer.json" in tokenizer_artifacts:
        return "tokenizer_json"
    if "vocab.json" in tokenizer_artifacts:
        return "bpe"
    return "unknown"


def _collect_special_tokens(config: dict[str, Any]) -> dict[str, Any]:
    token_keys = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "image_token_id",
        "video_token_id",
    )
    return {key: config[key] for key in token_keys if key in config}


def _shape_traits(tensor_infos: Iterable[Any]) -> list[str]:
    traits: list[str] = []
    for tensor in tensor_infos:
        if len(tensor.shape) == 3 and tensor.shape[-1] != 1 and (
            "conv1d.weight" in tensor.name or "conv_1d.weight" in tensor.name
        ):
            traits.append(f"conv_axis_last:{tensor.name}")
        if ".experts." in tensor.name:
            traits.append(f"expert_tensor:{tensor.name}")
    return traits


def _backbone_type(macro_template: MacroTemplate) -> str:
    return {
        MacroTemplate.DECODER_DENSE: "decoder",
        MacroTemplate.DECODER_MOE: "decoder",
        MacroTemplate.MULTIMODAL_DECODER: "multimodal_decoder",
        MacroTemplate.SSM_HYBRID: "ssm_hybrid",
        MacroTemplate.ENCODER_DECODER: "encoder_decoder",
        MacroTemplate.MULTIMODAL_ENCODER_DECODER: "multimodal_encoder_decoder",
    }[macro_template]


def _detect_mlp_variant(
    config: dict[str, Any],
    tensor_names: set[str],
    density: DensityKind,
) -> str | None:
    if density == DensityKind.MOE:
        return "switch_glu"
    if any("gelu" in str(config.get("hidden_act", "")).lower() for _ in [0]):
        return "gelu"
    if any("gate_proj" in name and "up_proj" in name for name in tensor_names):
        return "swiglu"
    return None


def _detect_norm_type(config: dict[str, Any]) -> str | None:
    if "rms_norm_eps" in config or "layer_norm_epsilon" in config:
        return "rmsnorm"
    return None


def _detect_rope_variant(config: dict[str, Any], tensor_names: set[str]) -> str | None:
    if "rope_scaling" in config or "rope_parameters" in config:
        return "scaled_rope"
    if any("rotary" in name or "rope" in name for name in tensor_names):
        return "rope"
    return None


def _value_from_nested(
    config: dict[str, Any],
    *keys: str,
    nested_key: str | None = None,
) -> Any:
    search_spaces: list[dict[str, Any]] = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        search_spaces.append(text_config)
    vision_config = config.get("vision_config") or config.get("visual_config")
    if isinstance(vision_config, dict):
        search_spaces.append(vision_config)
    if nested_key is not None:
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            search_spaces = [nested]
    for key in keys:
        for space in search_spaces:
            if key in space:
                return space[key]
    return None
