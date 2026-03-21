from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlxs._types import ModelMode
from mlxs.convert.errors import ConversionVerificationError
from mlxs.convert.runtime_schema import runtime_schema_hash
from mlxs.convert.types import (
    CanonicalIR,
    ConversionOptions,
    ConversionPhase,
    ConversionPlan,
    ExecutionResult,
    InspectionReport,
    RuntimeTensorSchemaEntry,
    VerificationCheck,
    VerificationMode,
    VerificationReport,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class _VerificationSpec:
    required_target_names: tuple[str, ...]
    target_schema: tuple[RuntimeTensorSchemaEntry, ...]
    target_schema_hash: str | None
    expected_artifacts: tuple[str, ...]
    expected_weight_files: tuple[str, ...]
    expected_weight_index_file: str | None
    expected_config: dict[str, Any]
    runtime_model_mode: str
    verification_policy: tuple[str, ...]


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
        return _skipped_report()

    spec = _spec_from_conversion(inspection, plan, execution)
    return _run_verification(spec, execution.output_dir, options=opts)


def verify_existing_output(
    output_dir: str | Path,
    *,
    options: ConversionOptions | None = None,
) -> VerificationReport:
    from mlxs.convert.inspection import inspect_source
    from mlxs.convert.normalization import normalize_inspection
    from mlxs.convert.planning import build_conversion_plan

    opts = options or ConversionOptions()
    if opts.verification_mode == VerificationMode.SKIP:
        return _skipped_report()

    output_path = Path(output_dir)
    inspection = inspect_source(output_path, options=opts)
    manifest = _load_manifest(output_path)

    if manifest is not None and _manifest_has_schema_snapshot(manifest):
        spec = _spec_from_manifest(
            output_path,
            inspection,
            manifest,
            verification_mode=opts.verification_mode,
        )
        return _run_verification(spec, output_path, options=opts)

    canonical_ir = normalize_inspection(inspection, options=opts)
    plan = build_conversion_plan(inspection, canonical_ir, options=opts)
    execution = _execution_from_output(output_path, inspection)
    return verify_conversion(
        inspection,
        canonical_ir,
        plan,
        execution,
        options=opts,
    )


def _skipped_report() -> VerificationReport:
    return VerificationReport(
        status=VerificationStatus.SKIPPED,
        checks=(
            VerificationCheck(
                "verification",
                VerificationStatus.SKIPPED,
                "verification skipped",
            ),
        ),
    )


def _spec_from_conversion(
    inspection: InspectionReport,
    plan: ConversionPlan,
    execution: ExecutionResult,
) -> _VerificationSpec:
    return _VerificationSpec(
        required_target_names=plan.required_target_names,
        target_schema=plan.target_schema,
        target_schema_hash=plan.target_schema_hash or runtime_schema_hash(plan.target_schema),
        expected_artifacts=tuple(
            sorted(set(inspection.tokenizer_artifacts) | set(inspection.multimodal_artifacts))
        ),
        expected_weight_files=execution.weight_files,
        expected_weight_index_file=execution.weight_index_file,
        expected_config=plan.normalized_config,
        runtime_model_mode=plan.runtime_model_mode,
        verification_policy=plan.verification_policy,
    )


def _spec_from_manifest(
    output_path: Path,
    inspection: InspectionReport,
    manifest: dict[str, Any],
    *,
    verification_mode: VerificationMode,
) -> _VerificationSpec:
    target_schema = _target_schema_from_snapshot(manifest["target_schema_snapshot"])
    expected_weight_files = (
        tuple(_string_tuple(manifest.get("weight_files"))) or inspection.shard_files
    )
    expected_weight_index_file = manifest.get("weight_index_file")
    if expected_weight_index_file is None and len(expected_weight_files) > 1:
        expected_weight_index_file = "model.safetensors.index.json"

    expected_config = manifest.get("normalized_config_snapshot")
    if not isinstance(expected_config, dict):
        expected_config = _load_output_config(output_path)

    expected_artifacts = tuple(
        sorted(
            set(_string_tuple(manifest.get("tokenizer_artifacts")))
            | set(_string_tuple(manifest.get("multimodal_artifacts")))
        )
    )
    target_schema_hash = manifest.get("target_schema_hash")
    include_schema_hash = isinstance(target_schema_hash, str) and bool(target_schema_hash)
    runtime_model_mode = str(manifest.get("runtime_model_mode") or "text")

    return _VerificationSpec(
        required_target_names=tuple(_string_tuple(manifest.get("required_target_names")))
        or tuple(entry.name for entry in target_schema),
        target_schema=target_schema,
        target_schema_hash=target_schema_hash if include_schema_hash else None,
        expected_artifacts=expected_artifacts,
        expected_weight_files=expected_weight_files,
        expected_weight_index_file=expected_weight_index_file,
        expected_config=expected_config,
        runtime_model_mode=runtime_model_mode,
        verification_policy=_default_policy_for_mode(
            verification_mode,
            include_schema_hash=include_schema_hash,
        ),
    )


def _execution_from_output(output_path: Path, inspection: InspectionReport) -> ExecutionResult:
    return ExecutionResult(
        output_dir=output_path,
        weight_files=inspection.shard_files,
        weight_index_file=(
            "model.safetensors.index.json"
            if (output_path / "model.safetensors.index.json").is_file()
            else None
        ),
        written_tensor_names=tuple(sorted(tensor.name for tensor in inspection.tensor_infos)),
        copied_artifacts=tuple(
            sorted(set(inspection.tokenizer_artifacts) | set(inspection.multimodal_artifacts))
        ),
        skipped_source_tensors=(),
    )


def _run_verification(
    spec: _VerificationSpec,
    output_dir: Path,
    *,
    options: ConversionOptions,
) -> VerificationReport:
    from mlxs.convert.inspection import inspect_source

    inspected = inspect_source(output_dir, options=options)
    checks: list[VerificationCheck] = []

    for check_name in spec.verification_policy:
        if check_name == "schema":
            _check_output_files(spec.expected_weight_files, output_dir, checks)
        elif check_name == "weight_index":
            _check_weight_index(spec.expected_weight_index_file, output_dir, checks)
        elif check_name == "config_invariants":
            _check_config_invariants(spec.expected_config, output_dir, checks)
        elif check_name == "required_tensor_coverage":
            _check_required_tensor_coverage(spec.required_target_names, inspected, checks)
        elif check_name == "shape":
            _check_tensor_shapes(spec.target_schema, inspected, checks)
        elif check_name == "schema_hash":
            _check_schema_hash(spec.target_schema_hash, inspected, checks)
        elif check_name == "artifacts":
            _check_artifact_completeness(spec.expected_artifacts, output_dir, checks)
        elif check_name == "runtime_smoke":
            _check_runtime_smoke_load(output_dir, spec.runtime_model_mode, checks)
        elif check_name == "minimal_forward":
            _check_minimal_forward(output_dir, spec.runtime_model_mode, checks)

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
            details={
                "checks": [
                    {"name": check.name, "detail": check.detail}
                    for check in checks
                    if check.status == VerificationStatus.FAILED
                ]
            },
        )
    return report


def _check_output_files(
    expected_weight_files: tuple[str, ...],
    output_dir: Path,
    checks: list[VerificationCheck],
) -> None:
    config_ok = (output_dir / "config.json").is_file()
    weight_ok = all((output_dir / file_name).is_file() for file_name in expected_weight_files)
    checks.append(
        VerificationCheck(
            "schema",
            VerificationStatus.PASSED if config_ok and weight_ok else VerificationStatus.FAILED,
            "config.json and expected safetensors files must exist",
        )
    )


def _check_weight_index(
    expected_weight_index_file: str | None,
    output_dir: Path,
    checks: list[VerificationCheck],
) -> None:
    if expected_weight_index_file is None:
        checks.append(
            VerificationCheck(
                "weight_index",
                VerificationStatus.PASSED,
                "no shard index required",
            )
        )
        return

    index_path = output_dir / expected_weight_index_file
    checks.append(
        VerificationCheck(
            "weight_index",
            VerificationStatus.PASSED if index_path.is_file() else VerificationStatus.FAILED,
            "sharded outputs include a safetensors index"
            if index_path.is_file()
            else f"missing shard index: {expected_weight_index_file}",
        )
    )


def _check_required_tensor_coverage(
    required_target_names: tuple[str, ...],
    inspected: InspectionReport,
    checks: list[VerificationCheck],
) -> None:
    written = {tensor.name for tensor in inspected.tensor_infos}
    required = set(required_target_names)
    missing = sorted(required - written)
    unexpected = sorted(written - required)
    passed = not missing and not unexpected
    detail = "all required target tensors were written exactly"
    if missing or unexpected:
        fragments: list[str] = []
        if missing:
            fragments.append(f"missing: {', '.join(missing[:10])}")
        if unexpected:
            fragments.append(f"unexpected: {', '.join(unexpected[:10])}")
        detail = "; ".join(fragments)
    checks.append(
        VerificationCheck(
            "required_tensor_coverage",
            VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            detail,
        )
    )


def _check_tensor_shapes(
    target_schema: tuple[RuntimeTensorSchemaEntry, ...],
    inspected: InspectionReport,
    checks: list[VerificationCheck],
) -> None:
    produced = {tensor.name: tensor.shape for tensor in inspected.tensor_infos}
    mismatches = []
    for entry in target_schema:
        actual_shape = produced.get(entry.name)
        if actual_shape is None:
            continue
        if tuple(actual_shape) != tuple(entry.shape):
            mismatches.append(entry.name)
    checks.append(
        VerificationCheck(
            "shape",
            VerificationStatus.PASSED if not mismatches else VerificationStatus.FAILED,
            "all produced tensor shapes match the expected schema"
            if not mismatches
            else f"shape mismatches: {', '.join(mismatches[:10])}",
        )
    )


def _check_schema_hash(
    expected_schema_hash: str | None,
    inspected: InspectionReport,
    checks: list[VerificationCheck],
) -> None:
    if expected_schema_hash is None:
        checks.append(
            VerificationCheck(
                "schema_hash",
                VerificationStatus.PASSED,
                "schema hash unavailable; check skipped",
            )
        )
        return

    produced_schema = tuple(
        RuntimeTensorSchemaEntry(
            name=tensor.name,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )
        for tensor in sorted(inspected.tensor_infos, key=lambda item: item.name)
    )
    actual_hash = runtime_schema_hash(produced_schema)
    checks.append(
        VerificationCheck(
            "schema_hash",
            (
                VerificationStatus.PASSED
                if actual_hash == expected_schema_hash
                else VerificationStatus.FAILED
            ),
            "produced schema hash matches the expected manifest/schema snapshot"
            if actual_hash == expected_schema_hash
            else (
                "schema hash mismatch: "
                f"expected {expected_schema_hash}, got {actual_hash}"
            ),
        )
    )


def _check_config_invariants(
    expected_config: dict[str, Any],
    output_dir: Path,
    checks: list[VerificationCheck],
) -> None:
    actual_config = _load_output_config(output_dir)
    if actual_config == expected_config:
        checks.append(
            VerificationCheck(
                "config_invariants",
                VerificationStatus.PASSED,
                "output config matches the normalized conversion config",
            )
        )
        return

    expected_keys = set(expected_config)
    actual_keys = set(actual_config)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    changed = sorted(
        key
        for key in expected_keys & actual_keys
        if actual_config.get(key) != expected_config.get(key)
    )
    fragments: list[str] = []
    if missing:
        fragments.append(f"missing keys: {', '.join(missing[:10])}")
    if extra:
        fragments.append(f"extra keys: {', '.join(extra[:10])}")
    if changed:
        fragments.append(f"changed keys: {', '.join(changed[:10])}")
    checks.append(
        VerificationCheck(
            "config_invariants",
            VerificationStatus.FAILED,
            "; ".join(fragments) or "normalized config mismatch",
        )
    )


def _check_artifact_completeness(
    expected_artifacts: tuple[str, ...],
    output_dir: Path,
    checks: list[VerificationCheck],
) -> None:
    present = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing = sorted(set(expected_artifacts) - present)
    checks.append(
        VerificationCheck(
            "artifacts",
            VerificationStatus.PASSED if not missing else VerificationStatus.FAILED,
            "tokenizer and multimodal artifacts copied"
            if not missing
            else f"missing artifacts: {', '.join(missing)}",
        )
    )


def _check_runtime_smoke_load(
    output_dir: Path,
    runtime_model_mode: str,
    checks: list[VerificationCheck],
) -> None:
    try:
        from mlxs.load.loader import load_model

        load_model(
            output_dir,
            lazy=False,
            model_mode=ModelMode(runtime_model_mode),
        )
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


def _check_minimal_forward(
    output_dir: Path,
    runtime_model_mode: str,
    checks: list[VerificationCheck],
) -> None:
    if runtime_model_mode != ModelMode.TEXT.value:
        checks.append(
            VerificationCheck(
                "minimal_forward",
                VerificationStatus.SKIPPED,
                "minimal forward is only enabled for text-mode runtime targets",
            )
        )
        return

    try:
        import mlx.core as mx

        from mlxs.load.loader import load_model

        model = load_model(
            output_dir,
            lazy=False,
            model_mode=ModelMode.TEXT,
        )
        logits = model(mx.array([[1, 2]], dtype=mx.int32))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        shape = tuple(int(dim) for dim in logits.shape)
    except Exception as exc:
        checks.append(
            VerificationCheck(
                "minimal_forward",
                VerificationStatus.FAILED,
                f"minimal forward failed: {exc}",
            )
        )
        return

    valid = len(shape) == 3 and shape[0] == 1 and shape[1] == 2
    checks.append(
        VerificationCheck(
            "minimal_forward",
            VerificationStatus.PASSED if valid else VerificationStatus.FAILED,
            "text-mode minimal forward produced logits with expected leading dimensions"
            if valid
            else f"unexpected minimal forward output shape: {shape}",
        )
    )


def _load_output_config(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    try:
        payload = json.loads(config_path.read_text())
    except OSError as exc:
        raise ConversionVerificationError(
            f"Unable to read config.json during verification: {exc}",
            phase=ConversionPhase.VERIFICATION,
            details={"config_path": str(config_path)},
        ) from exc
    if not isinstance(payload, dict):
        raise ConversionVerificationError(
            "config.json must contain an object payload",
            phase=ConversionPhase.VERIFICATION,
            details={"config_path": str(config_path)},
        )
    return payload


def _load_manifest(output_path: Path) -> dict[str, Any] | None:
    manifest_path = output_path / "conversion_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConversionVerificationError(
            f"Invalid conversion_manifest.json: {exc}",
            phase=ConversionPhase.VERIFICATION,
            details={"manifest_path": str(manifest_path)},
        ) from exc
    if not isinstance(payload, dict):
        raise ConversionVerificationError(
            "conversion_manifest.json must contain an object payload",
            phase=ConversionPhase.VERIFICATION,
            details={"manifest_path": str(manifest_path)},
        )
    return payload


def _manifest_has_schema_snapshot(manifest: dict[str, Any]) -> bool:
    snapshot = manifest.get("target_schema_snapshot")
    return isinstance(snapshot, (list, tuple)) and bool(snapshot)


def _target_schema_from_snapshot(snapshot: Any) -> tuple[RuntimeTensorSchemaEntry, ...]:
    if not isinstance(snapshot, (list, tuple)):
        raise ConversionVerificationError(
            "Manifest target_schema_snapshot must be a list",
            phase=ConversionPhase.VERIFICATION,
        )
    entries: list[RuntimeTensorSchemaEntry] = []
    for item in snapshot:
        if not isinstance(item, dict):
            raise ConversionVerificationError(
                "Manifest target_schema_snapshot entries must be objects",
                phase=ConversionPhase.VERIFICATION,
            )
        shape = item.get("shape")
        if not isinstance(shape, (list, tuple)):
            raise ConversionVerificationError(
                "Manifest schema entry shape must be a list",
                phase=ConversionPhase.VERIFICATION,
            )
        entries.append(
            RuntimeTensorSchemaEntry(
                name=str(item["name"]),
                shape=tuple(int(dim) for dim in shape),
                dtype=str(item["dtype"]) if item.get("dtype") is not None else None,
            )
        )
    return tuple(entries)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _default_policy_for_mode(
    mode: VerificationMode,
    *,
    include_schema_hash: bool,
) -> tuple[str, ...]:
    if mode == VerificationMode.SKIP:
        return ()
    basic = [
        "schema",
        "weight_index",
        "config_invariants",
        "required_tensor_coverage",
    ]
    if mode == VerificationMode.BASIC:
        return tuple(basic)

    policy = [*basic, "shape"]
    if include_schema_hash:
        policy.append("schema_hash")
    policy.append("artifacts")
    if mode in (VerificationMode.REQUIRED, VerificationMode.PARANOID):
        policy.append("runtime_smoke")
    if mode == VerificationMode.PARANOID:
        policy.append("minimal_forward")
    return tuple(policy)
