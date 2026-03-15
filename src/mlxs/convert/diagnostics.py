from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from mlxs.convert.errors import ConverterError
from mlxs.convert.types import (
    CanonicalIR,
    ConversionManifest,
    ConversionPlan,
    ConversionResult,
    ExecutionResult,
    InspectionReport,
    VerificationCheck,
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
    warnings = inspection.warnings
    if verification is not None:
        warnings = warnings + verification.warnings
    return ConversionManifest(
        source_identifier=inspection.source_id,
        macro_template=canonical_ir.identity.macro_template.value if canonical_ir else None,
        architecture_label=canonical_ir.identity.architecture_label if canonical_ir else None,
        runtime_target_model_type=(
            canonical_ir.identity.runtime_target_model_type if canonical_ir else None
        ),
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
        failure_phase=error.phase.value if error is not None else None,
        failure_reason=error.message if error is not None else None,
    )


def result_to_dict(result: ConversionResult) -> dict[str, Any]:
    return json_ready(result)
