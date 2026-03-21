from __future__ import annotations

from typing import Any

from mlxs._types import ModelMode
from mlxs.convert.errors import (
    ConverterError,
    MissingRequiredTensorError,
    UnsupportedRuntimeTargetError,
)
from mlxs.convert.runtime_schema import (
    _flatten_parameter_tree as _runtime_flatten_parameter_tree,
)
from mlxs.convert.runtime_schema import (
    export_runtime_schema,
    runtime_schema_hash,
)
from mlxs.convert.types import (
    CanonicalIR,
    ConversionOptions,
    ConversionPhase,
    ConversionPlan,
    InspectionReport,
    PlanningContext,
    RuntimeTensorSchemaEntry,
    TensorTargetPlan,
    TensorTransform,
    TensorTransformKind,
    VerificationMode,
)
from mlxs.family_adapters import AdapterAliasMatch, get_family_adapter


def build_conversion_plan(
    inspection: InspectionReport,
    canonical_ir: CanonicalIR,
    *,
    options: ConversionOptions | None = None,
) -> ConversionPlan:
    opts = options or ConversionOptions()
    if not canonical_ir.identity.supported_by_runtime:
        raise UnsupportedRuntimeTargetError(
            (
                "This repository snapshot does not provide a loadable runtime target for "
                f"{canonical_ir.identity.macro_template.value}"
            ),
            phase=ConversionPhase.PLANNING,
            details={"macro_template": canonical_ir.identity.macro_template.value},
        )

    target_schema = _collect_runtime_tensor_schema(canonical_ir)
    _ensure_unique_names(
        (entry.name for entry in target_schema),
        phase=ConversionPhase.PLANNING,
        noun="target schema tensor",
    )
    source_tensors = {tensor.name: tensor for tensor in inspection.tensor_infos}
    context = PlanningContext(
        source_tensors=source_tensors,
        target_schema={entry.name: entry for entry in target_schema},
    )
    mappings = tuple(_build_mappings_for_schema(context, canonical_ir))
    _ensure_unique_names(
        (mapping.target_name for mapping in mappings),
        phase=ConversionPhase.PLANNING,
        noun="mapped target",
    )
    missing = tuple(
        entry.name
        for entry in target_schema
        if entry.name not in {mapping.target_name for mapping in mappings}
    )
    if missing:
        raise MissingRequiredTensorError(
            f"Unable to map required tensors: {', '.join(missing[:10])}",
            phase=ConversionPhase.PLANNING,
            details={"missing_target_tensors": missing},
        )

    skipped_source = tuple(sorted(set(source_tensors) - context.used_source_names))
    return ConversionPlan(
        macro_template=canonical_ir.identity.macro_template,
        runtime_target_model_type=canonical_ir.identity.runtime_target_model_type,
        runtime_model_mode=_runtime_model_mode(canonical_ir),
        selected_profile=canonical_ir.tensor_layout.selected_source_tensor_profile,
        mappings=mappings,
        skipped_source_tensors=skipped_source,
        target_schema=tuple(target_schema),
        required_target_names=tuple(entry.name for entry in target_schema),
        selected_rules=canonical_ir.evidence.selected_rules,
        verification_policy=_verification_policy_for_mode(opts.verification_mode),
        normalized_config=canonical_ir.conversion.canonical_output_config,
        target_schema_hash=runtime_schema_hash(target_schema),
    )


def _collect_runtime_tensor_schema(canonical_ir: CanonicalIR) -> list[RuntimeTensorSchemaEntry]:
    return list(
        export_runtime_schema(
            canonical_ir.conversion.canonical_output_config,
            canonical_ir.identity.runtime_target_model_type,
            model_mode=_runtime_model_mode(canonical_ir),
        )
    )


def _flatten_parameter_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    return _runtime_flatten_parameter_tree(tree, prefix)


def _build_mappings_for_schema(
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> list[TensorTargetPlan]:
    mappings: list[TensorTargetPlan] = []
    for target_name, schema in context.target_schema.items():
        mapping = _mapping_for_target(target_name, schema.shape, context, canonical_ir)
        if mapping is not None:
            mappings.append(mapping)
    return mappings


def _mapping_for_target(
    target_name: str,
    target_shape: tuple[int, ...],
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> TensorTargetPlan | None:
    if exact := _mapping_from_exact_match(
        target_name,
        target_shape,
        context,
        canonical_ir,
    ):
        return exact

    if adapter_match := _mapping_from_family_adapter(
        target_name,
        target_shape,
        context,
        canonical_ir,
    ):
        return adapter_match

    specialized = (
        _stack_triplet_experts(target_name, context, canonical_ir)
        or _stack_named_experts(target_name, context, canonical_ir)
        or _split_switch_mlp_input_linear(target_name, context)
        or _split_gate_up_proj(target_name, context)
        or _split_shared_mlp(target_name, context)
        or _split_feed_forward_experts(target_name, context)
    )
    if specialized is None:
        return _mapping_from_generic_alias(
            target_name,
            target_shape,
            context,
            canonical_ir,
        )
    for source_name in specialized.source_names:
        context.used_source_names.add(source_name)
    return specialized


def _mapping_from_exact_match(
    target_name: str,
    target_shape: tuple[int, ...],
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> TensorTargetPlan | None:
    return _mapping_from_source_names(
        target_name,
        target_shape,
        source_names=(target_name,),
        context=context,
        canonical_ir=canonical_ir,
        rule_id="exact",
        match_layer="exact",
    )


def _mapping_from_family_adapter(
    target_name: str,
    target_shape: tuple[int, ...],
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> TensorTargetPlan | None:
    adapter = get_family_adapter(canonical_ir.identity.runtime_target_model_type)
    if adapter is None:
        return None
    model_mode = _planner_model_mode(canonical_ir)
    for match in adapter.alias_matches(target_name, model_mode=model_mode):
        mapping = _mapping_from_adapter_alias_match(
            target_name,
            target_shape,
            context,
            canonical_ir,
            match,
            adapter_name=adapter.adapter_name,
        )
        if mapping is not None:
            return mapping
    return None


def _mapping_from_generic_alias(
    target_name: str,
    target_shape: tuple[int, ...],
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> TensorTargetPlan | None:
    for candidate in _generic_alias_candidates(target_name):
        mapping = _mapping_from_source_names(
            target_name,
            target_shape,
            source_names=(candidate,),
            context=context,
            canonical_ir=canonical_ir,
            rule_id="generic_alias",
            match_layer="generic_alias",
            note=f"Matched via generic alias candidate {candidate}",
        )
        if mapping is not None:
            return mapping
    return None


def _mapping_from_adapter_alias_match(
    target_name: str,
    target_shape: tuple[int, ...],
    context: PlanningContext,
    canonical_ir: CanonicalIR,
    match: AdapterAliasMatch,
    *,
    adapter_name: str,
) -> TensorTargetPlan | None:
    return _mapping_from_source_names(
        target_name,
        target_shape,
        source_names=match.source_names,
        context=context,
        canonical_ir=canonical_ir,
        rule_id=match.rule_id,
        match_layer="family_adapter",
        adapter_name=adapter_name,
        note=match.note,
    )


def _mapping_from_source_names(
    target_name: str,
    target_shape: tuple[int, ...],
    *,
    source_names: tuple[str, ...],
    context: PlanningContext,
    canonical_ir: CanonicalIR,
    rule_id: str,
    match_layer: str,
    adapter_name: str | None = None,
    note: str | None = None,
) -> TensorTargetPlan | None:
    if len(source_names) != 1:
        if not all(name in context.source_tensors for name in source_names):
            return None
        for source_name in source_names:
            context.used_source_names.add(source_name)
        return TensorTargetPlan(
            target_name=target_name,
            source_names=source_names,
            rule_id=rule_id,
            match_layer=match_layer,
            adapter_name=adapter_name,
            note=note,
        )

    candidate = source_names[0]
    tensor = context.source_tensors.get(candidate)
    if tensor is None:
        return None
    transforms = list(
        _shape_adjustments(
            candidate,
            tensor.shape,
            target_name,
            target_shape,
            context,
            canonical_ir,
        )
    )
    context.used_source_names.add(candidate)
    return TensorTargetPlan(
        target_name=target_name,
        source_names=(candidate,),
        transforms=tuple(transforms),
        rule_id=rule_id,
        match_layer=match_layer,
        adapter_name=adapter_name,
        note=note,
    )


def _alias_candidates(target_name: str) -> tuple[str, ...]:
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(target_name)
    if target_name.startswith("language_model."):
        add(target_name[len("language_model."):])
    else:
        add(f"language_model.{target_name}")
    if target_name.startswith("language_model.model."):
        add(target_name[len("language_model."):])
        add(target_name.replace("language_model.model.", "model.", 1))
    if target_name.startswith("model."):
        add(target_name[len("model."):])
    if "vision_tower." in target_name:
        add(target_name.replace("vision_tower.", "visual.", 1))
        add(target_name.replace("vision_tower.", "model.visual.", 1))
        add(target_name.replace("vision_tower.", "model.vision_encoder.", 1))
        add(target_name.replace("vision_tower.", "vision_tower.vision_model.", 1))
    if "multi_modal_projector." in target_name:
        add(target_name.replace("multi_modal_projector.", "model.vision_projection.", 1))
    if target_name.startswith("language_model.model."):
        add(target_name.replace("language_model.model.", "model.language_model.", 1))
    for candidate in list(candidates):
        if ".mlp.fc1." in candidate:
            add(candidate.replace(".mlp.fc1.", ".mlp.linear_fc1.", 1))
        if ".mlp.fc2." in candidate:
            add(candidate.replace(".mlp.fc2.", ".mlp.linear_fc2.", 1))
        if ".merger.ln_q." in candidate:
            add(candidate.replace(".merger.ln_q.", ".merger.norm.", 1))
        if ".merger.mlp.0." in candidate:
            add(candidate.replace(".merger.mlp.0.", ".merger.linear_fc1.", 1))
        if ".merger.mlp.2." in candidate:
            add(candidate.replace(".merger.mlp.2.", ".merger.linear_fc2.", 1))
    return tuple(candidates)


def _generic_alias_candidates(target_name: str) -> tuple[str, ...]:
    return tuple(
        candidate for candidate in _alias_candidates(target_name) if candidate != target_name
    )


def _shape_adjustments(
    source_name: str,
    source_shape: tuple[int, ...],
    target_name: str,
    target_shape: tuple[int, ...],
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> list[TensorTransform]:
    transforms: list[TensorTransform] = []
    if _needs_conv_axis_move(source_name, source_shape, target_shape):
        transforms.append(
            TensorTransform(
                kind=TensorTransformKind.MOVE_AXIS,
                source_axis=2,
                target_axis=1,
                note="Normalize conv weight layout to runtime schema",
            )
        )
    elif permutation := _patch_conv_permutation(
        source_name,
        target_name,
        source_shape,
        target_shape,
    ):
        transforms.append(
            TensorTransform(
                kind=TensorTransformKind.TRANSPOSE,
                permutation=permutation,
                note="Normalize vision patch convolution layout to runtime schema",
            )
        )
    elif source_shape != target_shape and len(source_shape) == len(target_shape) and tuple(
        reversed(source_shape)
    ) == target_shape:
        transforms.append(
            TensorTransform(
                kind=TensorTransformKind.TRANSPOSE,
                permutation=tuple(range(len(source_shape) - 1, -1, -1)),
                note="Reverse axes to fit runtime schema",
            )
        )
    if _needs_qwen35_norm_shift(target_name, context, canonical_ir):
        transforms.append(
            TensorTransform(
                kind=TensorTransformKind.ADD,
                scalar=1.0,
                note="Restore runtime norm baseline after Qwen3.5-style source normalization",
            )
        )
    return transforms


def _needs_conv_axis_move(
    source_name: str,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> bool:
    if "conv1d.weight" not in source_name and "conv_1d.weight" not in source_name:
        return False
    if len(source_shape) != 3:
        return False
    moved = list(source_shape)
    moved[1], moved[2] = moved[2], moved[1]
    return tuple(moved) == target_shape


def _patch_conv_permutation(
    source_name: str,
    target_name: str,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    if not any(
        marker in source_name or marker in target_name
        for marker in ("patch_conv.weight", "patch_embed.proj.weight")
    ):
        return None
    if len(source_shape) not in {4, 5} or len(source_shape) != len(target_shape):
        return None
    permutation = (0, *range(2, len(source_shape)), 1)
    permuted = tuple(source_shape[index] for index in permutation)
    if permuted == target_shape:
        return tuple(int(index) for index in permutation)
    return None


def _stack_triplet_experts(
    target_name: str,
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> TensorTargetPlan | None:
    if ".switch_mlp." not in target_name:
        return None
    mapping = {
        "gate_proj": "w1",
        "down_proj": "w2",
        "up_proj": "w3",
    }
    matched = next((key for key in mapping if f".switch_mlp.{key}." in target_name), None)
    if matched is None:
        return None
    suffix = target_name.rsplit(".", 1)[-1]
    target_prefix = target_name.split(".switch_mlp.", 1)[0]
    source_stem = mapping[matched]
    expert_count = _expert_count(canonical_ir)
    if expert_count <= 0:
        return None
    source_names = tuple(
        f"{target_prefix}.experts.{index}.{source_stem}.{suffix}"
        for index in range(expert_count)
    )
    if not all(name in context.source_tensors for name in source_names):
        return None
    return TensorTargetPlan(
        target_name=target_name,
        source_names=source_names,
        transforms=(
            TensorTransform(
                kind=TensorTransformKind.STACK,
                axis=0,
                note="Stack per-expert triplets into runtime switch_mlp tensor",
            ),
        ),
        rule_id="stack_triplet_experts",
        match_layer="structural",
    )


def _stack_named_experts(
    target_name: str,
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> TensorTargetPlan | None:
    if ".switch_mlp." not in target_name:
        return None
    matched = next(
        (
            projection
            for projection in ("gate_proj", "down_proj", "up_proj")
            if f".switch_mlp.{projection}." in target_name
        ),
        None,
    )
    if matched is None:
        return None
    suffix = target_name.rsplit(".", 1)[-1]
    target_prefix = target_name.split(".switch_mlp.", 1)[0]
    expert_count = _expert_count(canonical_ir)
    if expert_count <= 0:
        return None
    source_names = tuple(
        f"{target_prefix}.experts.{index}.{matched}.{suffix}"
        for index in range(expert_count)
    )
    if not all(name in context.source_tensors for name in source_names):
        return None
    return TensorTargetPlan(
        target_name=target_name,
        source_names=source_names,
        transforms=(
            TensorTransform(
                kind=TensorTransformKind.STACK,
                axis=0,
                note="Stack per-expert tensors into runtime switch_mlp tensor",
            ),
        ),
        rule_id="stack_named_experts",
        match_layer="structural",
    )


def _split_switch_mlp_input_linear(
    target_name: str,
    context: PlanningContext,
) -> TensorTargetPlan | None:
    if ".switch_mlp.gate_proj.weight" in target_name:
        source_name = target_name.replace(
            ".switch_mlp.gate_proj.weight",
            ".input_linear.weight",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                transforms=(
                    TensorTransform(
                        kind=TensorTransformKind.SLICE,
                        axis=-2,
                        slice_start=0,
                        slice_stop=None,
                        note="Split switch MLP input_linear first half",
                    ),
                ),
                note="switch_mlp_input_linear_split",
                rule_id="switch_mlp_input_linear_split",
                match_layer="structural",
            )
    if ".switch_mlp.up_proj.weight" in target_name:
        source_name = target_name.replace(
            ".switch_mlp.up_proj.weight",
            ".input_linear.weight",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                transforms=(
                    TensorTransform(
                        kind=TensorTransformKind.SLICE,
                        axis=-2,
                        slice_start=None,
                        slice_stop=None,
                        note="Split switch MLP input_linear second half",
                    ),
                ),
                note="switch_mlp_input_linear_split",
                rule_id="switch_mlp_input_linear_split",
                match_layer="structural",
            )
    if ".switch_mlp.down_proj.weight" in target_name:
        source_name = target_name.replace(
            ".switch_mlp.down_proj.weight",
            ".output_linear.weight",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                rule_id="switch_mlp_output_linear_alias",
                match_layer="structural",
            )
    return None


def _split_gate_up_proj(target_name: str, context: PlanningContext) -> TensorTargetPlan | None:
    replacements = (
        (
            ".switch_mlp.gate_proj.weight",
            ".experts.gate_up_proj",
            True,
        ),
        (
            ".switch_mlp.up_proj.weight",
            ".experts.gate_up_proj",
            False,
        ),
    )
    for target_suffix, source_suffix, is_gate in replacements:
        if not target_name.endswith(target_suffix):
            continue
        source_name = _resolve_source_name(
            context,
            target_name[: -len(target_suffix)] + source_suffix,
            allow_weight_suffix=True,
        )
        if source_name is None:
            return None
        return TensorTargetPlan(
            target_name=target_name,
            source_names=(source_name,),
            transforms=(
                TensorTransform(
                    kind=TensorTransformKind.SLICE,
                    axis=-2,
                    slice_start=0 if is_gate else None,
                    slice_stop=None,
                    note=(
                        "Split combined gate_up expert tensor first half"
                        if is_gate
                        else "Split combined gate_up expert tensor second half"
                    ),
                ),
            ),
            note="gate_up_split",
            rule_id="gate_up_split",
            match_layer="structural",
        )
    if target_name.endswith(".switch_mlp.down_proj.weight"):
        source_name = _resolve_source_name(
            context,
            target_name[: -len(".switch_mlp.down_proj.weight")] + ".experts.down_proj",
            allow_weight_suffix=True,
        )
        if source_name is None:
            return None
        return TensorTargetPlan(
            target_name=target_name,
            source_names=(source_name,),
            note="expert_down_proj_alias",
            rule_id="expert_down_proj_alias",
            match_layer="structural",
        )
    return None


def _split_shared_mlp(target_name: str, context: PlanningContext) -> TensorTargetPlan | None:
    if ".mlp.gate_proj.weight" in target_name:
        source_name = target_name.replace(
            ".mlp.gate_proj.weight",
            ".shared_mlp.input_linear.weight",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                transforms=(
                    TensorTransform(
                        kind=TensorTransformKind.SLICE,
                        axis=0,
                        slice_start=0,
                        slice_stop=None,
                        note="Split shared MLP input_linear first half",
                    ),
                ),
                note="shared_mlp_split",
                rule_id="shared_mlp_split",
                match_layer="structural",
            )
    if ".mlp.up_proj.weight" in target_name:
        source_name = target_name.replace(
            ".mlp.up_proj.weight",
            ".shared_mlp.input_linear.weight",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                transforms=(
                    TensorTransform(
                        kind=TensorTransformKind.SLICE,
                        axis=0,
                        slice_start=None,
                        slice_stop=None,
                        note="Split shared MLP input_linear second half",
                    ),
                ),
                note="shared_mlp_split",
                rule_id="shared_mlp_split",
                match_layer="structural",
            )
    if ".mlp.down_proj.weight" in target_name:
        source_name = target_name.replace(
            ".mlp.down_proj.weight",
            ".shared_mlp.output_linear.weight",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                rule_id="shared_mlp_output_linear_alias",
                match_layer="structural",
            )
    return None


def _split_feed_forward_experts(
    target_name: str,
    context: PlanningContext,
) -> TensorTargetPlan | None:
    if ".feed_forward.experts.gate_proj.weight" in target_name:
        source_name = target_name.replace(
            ".feed_forward.experts.gate_proj.weight",
            ".feed_forward.experts.gate_up_proj",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                transforms=(
                    TensorTransform(
                        kind=TensorTransformKind.SLICE,
                        axis=-1,
                        slice_start=0,
                        slice_stop=None,
                        note="Split feed_forward expert gate half",
                    ),
                    TensorTransform(
                        kind=TensorTransformKind.MOVE_AXIS,
                        source_axis=1,
                        target_axis=2,
                        note="Align feed_forward expert gate weight axes",
                    ),
                ),
                rule_id="feed_forward_expert_gate_up_split",
                match_layer="structural",
            )
    if ".feed_forward.experts.up_proj.weight" in target_name:
        source_name = target_name.replace(
            ".feed_forward.experts.up_proj.weight",
            ".feed_forward.experts.gate_up_proj",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                transforms=(
                    TensorTransform(
                        kind=TensorTransformKind.SLICE,
                        axis=-1,
                        slice_start=None,
                        slice_stop=None,
                        note="Split feed_forward expert up half",
                    ),
                    TensorTransform(
                        kind=TensorTransformKind.MOVE_AXIS,
                        source_axis=1,
                        target_axis=2,
                        note="Align feed_forward expert up weight axes",
                    ),
                ),
                rule_id="feed_forward_expert_gate_up_split",
                match_layer="structural",
            )
    if ".feed_forward.experts.down_proj.weight" in target_name:
        source_name = target_name.replace(
            ".feed_forward.experts.down_proj.weight",
            ".feed_forward.experts.down_proj",
        )
        if source_name in context.source_tensors:
            return TensorTargetPlan(
                target_name=target_name,
                source_names=(source_name,),
                transforms=(
                    TensorTransform(
                        kind=TensorTransformKind.MOVE_AXIS,
                        source_axis=1,
                        target_axis=2,
                        note="Align feed_forward expert down weight axes",
                    ),
                ),
                rule_id="feed_forward_expert_down_align",
                match_layer="structural",
            )
    return None


def _expert_count(canonical_ir: CanonicalIR) -> int:
    value = canonical_ir.config.values.get("num_experts")
    if isinstance(value, int) and value > 0:
        return value
    return 0


def _resolve_source_name(
    context: PlanningContext,
    source_name: str,
    *,
    allow_weight_suffix: bool = False,
) -> str | None:
    candidates = list(_alias_candidates(source_name))
    if allow_weight_suffix:
        candidates.extend(
            candidate if candidate.endswith(".weight") else f"{candidate}.weight"
            for candidate in list(candidates)
        )
    for candidate in candidates:
        if candidate in context.source_tensors:
            return candidate
    return None


def _needs_qwen35_norm_shift(
    target_name: str,
    context: PlanningContext,
    canonical_ir: CanonicalIR,
) -> bool:
    if not _is_norm_target(target_name):
        return False
    tensor_names = context.source_tensors.keys()
    if not any("linear_attn." in name for name in tensor_names):
        return False
    if not any("self_attn." in name for name in tensor_names):
        return False
    if not any(
        ("conv1d.weight" in name or "conv_1d.weight" in name) and info.shape[-1] != 1
        for name, info in context.source_tensors.items()
    ):
        return False
    config = canonical_ir.conversion.canonical_output_config
    return (canonical_ir.topology.ssm_hybrid or canonical_ir.topology.multimodal) and (
        _config_contains(config, "partial_rotary_factor") or _config_contains(
        config,
        "rope_parameters",
    )
    )


def _is_norm_target(target_name: str) -> bool:
    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    return any(target_name.endswith(suffix) for suffix in norm_suffixes)


def _config_contains(config: dict[str, Any], key: str) -> bool:
    if key in config:
        return True
    for value in config.values():
        if isinstance(value, dict) and _config_contains(value, key):
            return True
    return False


def _runtime_model_mode(canonical_ir: CanonicalIR) -> str:
    if canonical_ir.topology.multimodal:
        return "multimodal"
    return "text"


def _planner_model_mode(canonical_ir: CanonicalIR) -> ModelMode:
    return ModelMode(_runtime_model_mode(canonical_ir))


def _verification_policy_for_mode(mode: VerificationMode) -> tuple[str, ...]:
    if mode == VerificationMode.SKIP:
        return ()
    base = (
        "schema",
        "weight_index",
        "config_invariants",
        "required_tensor_coverage",
        "shape",
        "schema_hash",
        "artifacts",
    )
    if mode == VerificationMode.REQUIRED:
        return (*base, "runtime_smoke")
    return base


def _ensure_unique_names(
    names: Any,
    *,
    phase: ConversionPhase,
    noun: str,
) -> None:
    ordered = [str(name) for name in names]
    duplicates = sorted({name for name in ordered if ordered.count(name) > 1})
    if duplicates:
        raise ConverterError(
            f"Duplicate {noun} names are not allowed: {', '.join(duplicates[:10])}",
            phase=phase,
            details={"duplicates": tuple(duplicates), "noun": noun},
        )
