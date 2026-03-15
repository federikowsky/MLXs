from __future__ import annotations

from pathlib import Path

from mlxs.convert.errors import ConversionVerificationError
from mlxs.convert.types import (
    CanonicalIR,
    ConversionOptions,
    ConversionPhase,
    ConversionPlan,
    ExecutionResult,
    InspectionReport,
    VerificationCheck,
    VerificationMode,
    VerificationReport,
    VerificationStatus,
)


def verify_conversion(
    inspection: InspectionReport,
    canonical_ir: CanonicalIR,
    plan: ConversionPlan,
    execution: ExecutionResult,
    *,
    options: ConversionOptions | None = None,
) -> VerificationReport:
    opts = options or ConversionOptions()
    if opts.verification_mode == VerificationMode.SKIP:
        return VerificationReport(
            status=VerificationStatus.SKIPPED,
            checks=(VerificationCheck("verification", VerificationStatus.SKIPPED, "verification skipped"),),
        )

    checks: list[VerificationCheck] = []
    _check_output_files(execution, checks)
    _check_required_tensor_coverage(plan, execution, checks)
    _check_tensor_shapes(plan, execution.output_dir, checks)
    _check_artifact_completeness(inspection, execution.output_dir, checks)
    if opts.verification_mode == VerificationMode.REQUIRED:
        _check_runtime_smoke_load(canonical_ir, execution.output_dir, checks)

    status = (
        VerificationStatus.PASSED
        if all(check.status != VerificationStatus.FAILED for check in checks)
        else VerificationStatus.FAILED
    )
    report = VerificationReport(status=status, checks=tuple(checks))
    if status == VerificationStatus.FAILED:
        raise ConversionVerificationError(
            "Verification failed for converted output",
            phase=ConversionPhase.VERIFICATION,
            details={"checks": [check.detail for check in checks if check.status == VerificationStatus.FAILED]},
        )
    return report


def verify_existing_output(
    output_dir: str | Path,
    *,
    options: ConversionOptions | None = None,
) -> VerificationReport:
    from mlxs.convert.inspection import inspect_source
    from mlxs.convert.normalization import normalize_inspection
    from mlxs.convert.planning import build_conversion_plan

    inspection = inspect_source(output_dir, options=options)
    canonical_ir = normalize_inspection(inspection, options=options)
    plan = build_conversion_plan(inspection, canonical_ir, options=options)
    execution = ExecutionResult(
        output_dir=Path(output_dir),
        weight_files=inspection.shard_files,
        weight_index_file=(
            "model.safetensors.index.json"
            if (Path(output_dir) / "model.safetensors.index.json").is_file()
            else None
        ),
        written_tensor_names=tuple(sorted(tensor.name for tensor in inspection.tensor_infos)),
        copied_artifacts=inspection.tokenizer_artifacts + inspection.multimodal_artifacts,
        skipped_source_tensors=(),
    )
    return verify_conversion(inspection, canonical_ir, plan, execution, options=options)


def _check_output_files(execution: ExecutionResult, checks: list[VerificationCheck]) -> None:
    config_ok = (execution.output_dir / "config.json").is_file()
    weight_ok = all((execution.output_dir / file_name).is_file() for file_name in execution.weight_files)
    checks.append(
        VerificationCheck(
            "schema",
            VerificationStatus.PASSED if config_ok and weight_ok else VerificationStatus.FAILED,
            "config.json and safetensors files must exist",
        )
    )


def _check_required_tensor_coverage(
    plan: ConversionPlan,
    execution: ExecutionResult,
    checks: list[VerificationCheck],
) -> None:
    written = set(execution.written_tensor_names)
    required = set(plan.required_target_names)
    missing = sorted(required - written)
    checks.append(
        VerificationCheck(
            "required_tensor_coverage",
            VerificationStatus.PASSED if not missing else VerificationStatus.FAILED,
            "all required target tensors must be written"
            if not missing
            else f"missing required tensors: {', '.join(missing[:10])}",
        )
    )


def _check_tensor_shapes(
    plan: ConversionPlan,
    output_dir: Path,
    checks: list[VerificationCheck],
) -> None:
    from mlxs.convert.inspection import inspect_source

    inspected = inspect_source(output_dir)
    produced = {tensor.name: tensor.shape for tensor in inspected.tensor_infos}
    mismatches = []
    for entry in plan.target_schema:
        if produced.get(entry.name) != entry.shape:
            mismatches.append(entry.name)
    checks.append(
        VerificationCheck(
            "shape",
            VerificationStatus.PASSED if not mismatches else VerificationStatus.FAILED,
            "all produced tensor shapes match the runtime schema"
            if not mismatches
            else f"shape mismatches: {', '.join(mismatches[:10])}",
        )
    )


def _check_artifact_completeness(
    inspection: InspectionReport,
    output_dir: Path,
    checks: list[VerificationCheck],
) -> None:
    required = set(inspection.tokenizer_artifacts)
    present = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing = sorted(required - present)
    checks.append(
        VerificationCheck(
            "artifacts",
            VerificationStatus.PASSED if not missing else VerificationStatus.FAILED,
            "tokenizer artifacts copied"
            if not missing
            else f"missing tokenizer artifacts: {', '.join(missing)}",
        )
    )


def _check_runtime_smoke_load(
    canonical_ir: CanonicalIR,
    output_dir: Path,
    checks: list[VerificationCheck],
) -> None:
    try:
        from mlxs.load.loader import load_model

        if canonical_ir.topology.multimodal:
            from mlxs._types import ModelMode

            load_model(output_dir, lazy=True, model_mode=ModelMode.MULTIMODAL)
        else:
            load_model(output_dir, lazy=True)
    except Exception as exc:
        checks.append(
            VerificationCheck(
                "runtime_smoke",
                VerificationStatus.FAILED,
                f"runtime smoke load failed: {exc}",
            )
        )
        return
    checks.append(
        VerificationCheck(
            "runtime_smoke",
            VerificationStatus.PASSED,
            "converted package loads through the current runtime loader",
        )
    )
