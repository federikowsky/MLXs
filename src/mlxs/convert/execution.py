from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlxs.convert.diagnostics import write_manifest
from mlxs.convert.errors import (
    IncompatibleTensorShapeError,
    InvalidTransformRequestError,
    SerializationFailureError,
)
from mlxs.convert.types import (
    CanonicalIR,
    ConversionManifest,
    ConversionOptions,
    ConversionPhase,
    ConversionPlan,
    ExecutionResult,
    InspectionReport,
    TensorTransformKind,
)


def execute_conversion(
    inspection: InspectionReport,
    canonical_ir: CanonicalIR,
    plan: ConversionPlan,
    output_dir: str | Path,
    *,
    options: ConversionOptions | None = None,
) -> ExecutionResult:
    opts = options or ConversionOptions()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    source_weights = _load_source_weights(inspection)
    written = _materialize_target_tensors(source_weights, plan)
    _write_config(destination, canonical_ir.conversion.canonical_output_config)
    weight_files, index_file = _write_weight_files(destination, written, opts.max_shard_bytes)
    copied_artifacts = _copy_artifacts(inspection, destination, copy_enabled=opts.copy_tokenizer_artifacts)

    return ExecutionResult(
        output_dir=destination,
        weight_files=tuple(weight_files),
        weight_index_file=index_file,
        written_tensor_names=tuple(sorted(written)),
        copied_artifacts=tuple(copied_artifacts),
        skipped_source_tensors=plan.skipped_source_tensors,
    )


def write_failure_manifest(output_dir: Path, manifest: ConversionManifest) -> Path:
    return write_manifest(output_dir / "conversion_manifest.json", manifest)


def _load_source_weights(inspection: InspectionReport) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for file_name in inspection.shard_files:
        file_path = inspection.resolved_path / file_name
        tensors.update(mx.load(str(file_path)))
    return tensors


def _materialize_target_tensors(
    source_weights: dict[str, Any],
    plan: ConversionPlan,
) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    target_shapes = {entry.name: entry.shape for entry in plan.target_schema}
    for mapping in plan.mappings:
        materialized = _apply_mapping(source_weights, mapping)
        expected_shape = target_shapes[mapping.target_name]
        actual_shape = tuple(int(dim) for dim in materialized.shape)
        if actual_shape != expected_shape:
            raise IncompatibleTensorShapeError(
                (
                    f"Produced shape {actual_shape} for {mapping.target_name}, "
                    f"expected {expected_shape}"
                ),
                phase=ConversionPhase.EXECUTION,
                details={
                    "target_name": mapping.target_name,
                    "expected_shape": expected_shape,
                    "actual_shape": actual_shape,
                },
            )
        tensors[mapping.target_name] = materialized
    return tensors


def _apply_mapping(source_weights: dict[str, Any], mapping: Any) -> Any:
    inputs = [source_weights[name] for name in mapping.source_names]
    value: Any = inputs[0] if len(inputs) == 1 else inputs
    for transform in mapping.transforms:
        value = _apply_transform(value, transform)
    return value


def _apply_transform(value: Any, transform: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise SerializationFailureError(
            "numpy is required to execute conversion transforms",
            phase=ConversionPhase.EXECUTION,
        ) from exc

    backend = _transform_backend(value)

    if transform.kind == TensorTransformKind.STACK:
        if not isinstance(value, list):
            raise InvalidTransformRequestError(
                "STACK transform expects multiple tensors",
                phase=ConversionPhase.EXECUTION,
            )
        if backend == "mlx":
            return mx.stack(value, axis=transform.axis or 0)
        return np.stack(value, axis=transform.axis or 0)

    if transform.kind == TensorTransformKind.CONCAT:
        if not isinstance(value, list):
            raise InvalidTransformRequestError(
                "CONCAT transform expects multiple tensors",
                phase=ConversionPhase.EXECUTION,
            )
        if backend == "mlx":
            return mx.concatenate(value, axis=transform.axis or 0)
        return np.concatenate(value, axis=transform.axis or 0)

    if transform.kind == TensorTransformKind.ADD:
        scalar = 0.0 if transform.scalar is None else transform.scalar
        if backend == "mlx":
            return value + scalar
        return value + np.asarray(scalar, dtype=value.dtype)

    if transform.kind == TensorTransformKind.MOVE_AXIS:
        if backend == "mlx":
            return mx.moveaxis(value, transform.source_axis, transform.target_axis)
        return np.moveaxis(value, transform.source_axis, transform.target_axis)

    if transform.kind == TensorTransformKind.TRANSPOSE:
        if backend == "mlx":
            return mx.transpose(value, axes=transform.permutation)
        return np.transpose(value, axes=transform.permutation)

    if transform.kind == TensorTransformKind.RESHAPE:
        if backend == "mlx":
            return mx.reshape(value, shape=transform.shape)
        return np.reshape(value, newshape=transform.shape)

    if transform.kind == TensorTransformKind.CAST:
        if backend == "mlx":
            return value.astype(_mlx_dtype(transform.dtype))
        return value.astype(transform.dtype)

    if transform.kind == TensorTransformKind.SLICE:
        axis = transform.axis if transform.axis is not None else 0
        size = value.shape[axis]
        midpoint = size // 2
        start = transform.slice_start
        stop = transform.slice_stop
        if start is None and stop is None:
            if "second" in (transform.note or "") or "up half" in (transform.note or ""):
                start = midpoint
                stop = size
            else:
                start = 0
                stop = midpoint
        elif start is None:
            start = midpoint
        elif stop is None:
            stop = midpoint
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(start, stop)
        return value[tuple(slices)]

    raise InvalidTransformRequestError(
        f"Unsupported transform: {transform.kind.value}",
        phase=ConversionPhase.EXECUTION,
    )


def _write_config(destination: Path, config: dict[str, Any]) -> None:
    try:
        (destination / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    except OSError as exc:
        raise SerializationFailureError(
            f"Failed to write config.json: {exc}",
            phase=ConversionPhase.EXECUTION,
        ) from exc


def _write_weight_files(
    destination: Path,
    tensors: dict[str, Any],
    max_shard_bytes: int | None,
) -> tuple[list[str], str | None]:
    if max_shard_bytes is None or not tensors:
        file_name = "model.safetensors"
        mx.save_safetensors(str(destination / file_name), _to_mlx_tensors(tensors))
        return [file_name], None

    shards: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_size = 0
    weight_map: dict[str, str] = {}
    for name, tensor in sorted(tensors.items()):
        tensor_size = int(getattr(tensor, "nbytes", 0))
        if current and current_size + tensor_size > max_shard_bytes:
            shards.append(current)
            current = {}
            current_size = 0
        current[name] = tensor
        current_size += tensor_size
    if current:
        shards.append(current)

    weight_files: list[str] = []
    total = len(shards)
    for index, shard in enumerate(shards, start=1):
        file_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        mx.save_safetensors(str(destination / file_name), _to_mlx_tensors(shard))
        weight_files.append(file_name)
        for name in shard:
            weight_map[name] = file_name

    index_name = "model.safetensors.index.json"
    index_payload = {"metadata": {"total_size": sum(int(t.nbytes) for t in tensors.values())}, "weight_map": weight_map}
    (destination / index_name).write_text(json.dumps(index_payload, indent=2, sort_keys=True))
    return weight_files, index_name


def _transform_backend(value: Any) -> str:
    sample = value[0] if isinstance(value, list) and value else value
    module = type(sample).__module__
    return "mlx" if module.startswith("mlx.") else "numpy"


def _to_mlx_tensors(tensors: dict[str, Any]) -> dict[str, mx.array]:
    result: dict[str, mx.array] = {}
    for name, value in tensors.items():
        result[name] = value if _transform_backend(value) == "mlx" else mx.array(value)
    return result


def _mlx_dtype(dtype_name: str | None) -> Any:
    if dtype_name is None:
        return None
    name = str(dtype_name).split(".")[-1]
    if hasattr(mx, name):
        return getattr(mx, name)
    return dtype_name


def _copy_artifacts(
    inspection: InspectionReport,
    destination: Path,
    *,
    copy_enabled: bool,
) -> list[str]:
    copied: list[str] = []
    if not copy_enabled:
        return copied
    artifact_names = set(inspection.tokenizer_artifacts) | set(inspection.multimodal_artifacts)
    for artifact_name in sorted(artifact_names):
        source_path = inspection.resolved_path / artifact_name
        if not source_path.is_file():
            continue
        shutil.copy2(source_path, destination / artifact_name)
        copied.append(artifact_name)
    return copied
