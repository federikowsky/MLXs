from __future__ import annotations

from pathlib import Path

from mlxs.convert.diagnostics import build_manifest
from mlxs.convert.execution import execute_conversion, write_failure_manifest
from mlxs.convert.errors import ConverterError
from mlxs.convert.inspection import inspect_source as _inspect_source
from mlxs.convert.normalization import normalize_inspection
from mlxs.convert.planning import build_conversion_plan
from mlxs.convert.types import ConversionOptions, ConversionResult
from mlxs.convert.verification import verify_conversion, verify_existing_output


def inspect_source(
    source: str | Path,
    *,
    options: ConversionOptions | None = None,
):
    return _inspect_source(source, options=options)


def convert_source(
    source: str | Path,
    output_dir: str | Path,
    *,
    options: ConversionOptions | None = None,
) -> ConversionResult:
    opts = options or ConversionOptions()
    output_path = Path(output_dir)
    inspection = _inspect_source(source, options=opts)
    canonical_ir = None
    plan = None
    execution = None
    verification = None
    try:
        canonical_ir = normalize_inspection(inspection, options=opts)
        plan = build_conversion_plan(inspection, canonical_ir, options=opts)
        execution = execute_conversion(
            inspection,
            canonical_ir,
            plan,
            output_path,
            options=opts,
        )
        verification = verify_conversion(
            inspection,
            canonical_ir,
            plan,
            execution,
            options=opts,
        )
    except ConverterError as exc:
        manifest = build_manifest(
            inspection,
            canonical_ir,
            plan,
            execution,
            verification,
            error=exc,
        )
        write_failure_manifest(output_path, manifest)
        raise
    except Exception as exc:
        wrapped = _wrap_unexpected_error(
            exc,
            canonical_ir=canonical_ir,
            plan=plan,
            execution=execution,
            verification=verification,
        )
        manifest = build_manifest(
            inspection,
            canonical_ir,
            plan,
            execution,
            verification,
            error=wrapped,
        )
        write_failure_manifest(output_path, manifest)
        raise wrapped from exc

    manifest = build_manifest(inspection, canonical_ir, plan, execution, verification)
    manifest_path = write_failure_manifest(output_path, manifest)
    return ConversionResult(
        inspection=inspection,
        canonical_ir=canonical_ir,
        plan=plan,
        execution=execution,
        verification=verification,
        manifest_path=manifest_path,
    )


def verify_output(
    output_dir: str | Path,
    *,
    options: ConversionOptions | None = None,
):
    try:
        return verify_existing_output(output_dir, options=options)
    except ConverterError:
        raise
    except Exception as exc:
        raise _wrap_unexpected_error(
            exc,
            canonical_ir=None,
            plan=None,
            execution=None,
            verification=None,
        ) from exc


def _wrap_unexpected_error(
    exc: Exception,
    *,
    canonical_ir,
    plan,
    execution,
    verification,
) -> ConverterError:
    if canonical_ir is None:
        phase = "normalization"
    elif plan is None:
        phase = "planning"
    elif execution is None:
        phase = "execution"
    else:
        phase = "verification"

    from mlxs.convert.types import ConversionPhase

    return ConverterError(
        f"Unexpected {phase} failure: {exc}",
        phase=ConversionPhase(phase),
        details={
            "exception_type": type(exc).__name__,
            "raw_message": str(exc),
        },
    )
