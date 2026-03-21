from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from mlxs.convert.errors import ConverterError
from mlxs.convert.runtime_schema import runtime_schema_hash
from mlxs.convert.types import (
    CanonicalIR,
    ConversionManifest,
    ConversionPlan,
    ConversionResult,
    ExecutionResult,
    InspectionReport,
    VerificationReport,
)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return value


def write_manifest(path: Path, manifest: ConversionManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True))
    return path


def build_manifest(
    inspection: InspectionReport,
    canonical_ir: CanonicalIR | None,
    plan: ConversionPlan | None,
    execution: ExecutionResult | None,
    verification: VerificationReport | None,
    *,
    error: ConverterError | None = None,
) -> ConversionManifest:
    checks = verification.checks if verification is not None else ()
    mapped_count = len(plan.mappings) if plan is not None else 0
    required_count = len(plan.required_target_names) if plan is not None else 0
    skipped = len(execution.skipped_source_tensors) if execution is not None else 0
    missing = max(required_count - mapped_count, 0)
    transforms: list[str] = []
    if plan is not None:
        for mapping in plan.mappings:
            for transform in mapping.transforms:
                transforms.append(transform.kind.value)

    structural_parameters = {}
    if canonical_ir is not None:
        structural_parameters = {
            "backbone_type": canonical_ir.topology.backbone_type,
            "density": canonical_ir.topology.density.value,
            "attention_variant": canonical_ir.topology.attention_variant,
            "qkv_layout": canonical_ir.topology.qkv_layout,
            "norm_type": canonical_ir.topology.norm_type,
            "rope_variant": canonical_ir.topology.rope_variant,
            "projector_presence": canonical_ir.topology.projector_presence,
            "tower_presence": canonical_ir.topology.tower_presence,
        }
        structural_parameters.update(canonical_ir.config.values)

    verification_checks = tuple(
        {
            "name": check.name,
            "status": check.status.value,
            "detail": check.detail,
        }
        for check in checks
    )
    if not verification_checks and error is not None and error.phase.value == "verification":
        failed_checks = error.details.get("checks")
        if isinstance(failed_checks, list):
            verification_checks = tuple(
                {
                    "name": str(item.get("name", "verification")),
                    "status": "failed",
                    "detail": str(item.get("detail", "")),
                }
                for item in failed_checks
                if isinstance(item, dict)
            )
    capability_snapshot: dict[str, Any] = {}
    architecture_traits: dict[str, Any] = {}
    execution_dependency_summary: dict[str, Any] = {}
    if canonical_ir is not None:
        capability_snapshot = _capability_snapshot(
            canonical_ir.identity.runtime_target_model_type
        )
        architecture_traits = json_ready(canonical_ir.traits)
    if execution is not None and execution.dependency_plan is not None:
        execution_dependency_summary = {
            "load_strategy": execution.load_strategy,
            "materialization_strategy": execution.materialization_strategy,
            "referenced_source_tensor_count": len(
                execution.dependency_plan.referenced_source_tensors
            ),
            "loaded_source_tensor_count": len(execution.loaded_source_tensors),
            "referenced_source_shards": execution.dependency_plan.referenced_source_shards,
            "source_shard_dependencies": tuple(
                {
                    "shard_file": group.shard_file,
                    "source_tensor_count": len(group.source_names),
                    "target_count": len(group.target_names),
                }
                for group in execution.dependency_plan.source_shard_dependencies
            ),
            "mapping_dependency_groups": tuple(
                {
                    "shard_files": group.shard_files,
                    "target_count": len(group.target_names),
                }
                for group in execution.dependency_plan.mapping_dependency_groups
            ),
            "output_pack_groups": tuple(
                {
                    "target_count": len(group.target_names),
                    "total_bytes": group.total_bytes,
                }
                for group in execution.output_pack_groups
            ),
        }

    mapping_provenance: tuple[dict[str, Any], ...] = ()
    target_schema_snapshot: tuple[dict[str, Any], ...] = ()
    target_schema_hash: str | None = None
    if plan is not None:
        mapping_provenance = tuple(
            {
                "target_name": mapping.target_name,
                "source_names": mapping.source_names,
                "rule_id": mapping.rule_id,
                "match_layer": mapping.match_layer,
                "adapter_name": mapping.adapter_name,
                "required": mapping.required,
                "note": mapping.note,
                "transforms": tuple(json_ready(transform) for transform in mapping.transforms),
            }
            for mapping in plan.mappings
        )
        target_schema_snapshot = tuple(
            {
                "name": entry.name,
                "shape": tuple(entry.shape),
                "dtype": entry.dtype,
            }
            for entry in plan.target_schema
        )
        target_schema_hash = plan.target_schema_hash or runtime_schema_hash(plan.target_schema)

    tokenizer_artifacts = tuple(getattr(inspection, "tokenizer_artifacts", ()))
    multimodal_artifacts = tuple(getattr(inspection, "multimodal_artifacts", ()))
    warnings = tuple(getattr(inspection, "warnings", ()))
    if verification is not None:
        warnings = warnings + verification.warnings
    return ConversionManifest(
        source_identifier=inspection.source_id,
        macro_template=canonical_ir.identity.macro_template.value if canonical_ir else None,
        architecture_label=canonical_ir.identity.architecture_label if canonical_ir else None,
        runtime_target_model_type=(
            canonical_ir.identity.runtime_target_model_type if canonical_ir else None
        ),
        runtime_model_mode=plan.runtime_model_mode if plan is not None else None,
        structural_parameters=structural_parameters,
        model_assisted_normalization_used=bool(
            canonical_ir
            and canonical_ir.evidence.model_assisted_normalization_usage is not None
        ),
        tensor_profile=plan.selected_profile if plan is not None else None,
        required_tensor_count=required_count,
        mapped_tensor_count=mapped_count,
        skipped_tensor_count=skipped,
        missing_tensor_count=missing,
        transforms=tuple(sorted(set(transforms))),
        output_format_version=(
            canonical_ir.conversion.target_format_version if canonical_ir else None
        ),
        quantization_mode=canonical_ir.conversion.quantization_mode if canonical_ir else None,
        verification_status=(
            verification.status.value if verification is not None else "not_run"
        ),
        verification_checks=verification_checks,
        warnings=warnings,
        architecture_traits=architecture_traits,
        execution_dependency_summary=execution_dependency_summary,
        capability_snapshot=capability_snapshot,
        required_target_names=plan.required_target_names if plan is not None else (),
        mapping_provenance=mapping_provenance,
        skipped_source_tensors=(
            execution.skipped_source_tensors if execution is not None else ()
        ),
        verification_policy=plan.verification_policy if plan is not None else (),
        target_schema_hash=target_schema_hash,
        target_schema_snapshot=target_schema_snapshot,
        normalized_config_snapshot=plan.normalized_config if plan is not None else {},
        tokenizer_artifacts=tokenizer_artifacts,
        multimodal_artifacts=multimodal_artifacts,
        copied_artifacts=execution.copied_artifacts if execution is not None else (),
        weight_files=execution.weight_files if execution is not None else (),
        weight_index_file=execution.weight_index_file if execution is not None else None,
        failure_code=error.code if error is not None else None,
        failure_phase=error.phase.value if error is not None else None,
        failure_reason=error.message if error is not None else None,
        failure_details=error.details if error is not None else {},
    )


def result_to_dict(result: ConversionResult) -> dict[str, Any]:
    return json_ready(result)


def _capability_snapshot(runtime_target_model_type: str) -> dict[str, Any]:
    from mlxs.load.registry import get_model_capabilities

    try:
        capabilities = get_model_capabilities(runtime_target_model_type)
    except ValueError:
        return {}
    return {
        "supports_text": capabilities.supports_text,
        "supports_multimodal": capabilities.supports_multimodal,
        "supported_model_modes": tuple(
            mode.value
            for mode in sorted(
                capabilities.supported_model_modes,
                key=lambda item: item.value,
            )
        ),
        "supports_conversion": capabilities.supports_conversion,
        "supports_schema_export": capabilities.supports_schema_export,
        "constructor_accepts_model_mode": capabilities.constructor_accepts_model_mode,
    }
