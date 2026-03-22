from __future__ import annotations

import json
import shutil
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
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
    ExecutionDependencyPlan,
    ExecutionResult,
    InspectionReport,
    MappingDependencyGroup,
    OutputPackGroup,
    SourceShardDependency,
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

    dependencies = _prepare_execution_dependencies(inspection, plan)
    _write_config(destination, canonical_ir.conversion.canonical_output_config)
    write_outcome = _execute_weight_write(
        inspection,
        plan,
        destination,
        dependencies=dependencies,
        max_shard_bytes=opts.max_shard_bytes,
    )
    copied_artifacts = _copy_artifacts(
        inspection,
        destination,
        copy_enabled=opts.copy_tokenizer_artifacts,
    )

    return ExecutionResult(
        output_dir=destination,
        weight_files=write_outcome.weight_files,
        weight_index_file=write_outcome.weight_index_file,
        written_tensor_names=write_outcome.written_tensor_names,
        copied_artifacts=tuple(copied_artifacts),
        skipped_source_tensors=plan.skipped_source_tensors,
        loaded_source_tensors=write_outcome.loaded_source_tensors,
        load_strategy=write_outcome.load_strategy,
        materialization_strategy=write_outcome.materialization_strategy,
        dependency_plan=dependencies,
        output_pack_groups=write_outcome.output_pack_groups,
    )


def write_failure_manifest(output_dir: Path, manifest: ConversionManifest) -> Path:
    return write_manifest(output_dir / "conversion_manifest.json", manifest)


def _prepare_execution_dependencies(
    inspection: InspectionReport,
    plan: ConversionPlan,
) -> ExecutionDependencyPlan:
    tensor_file_by_name = {tensor.name: tensor.file for tensor in inspection.tensor_infos}
    referenced_source_tensors = tuple(
        sorted(
            {
                source_name
                for mapping in plan.mappings
                for source_name in mapping.source_names
            }
        )
    )

    shard_to_sources: dict[str, list[str]] = {}
    shard_to_targets: dict[str, list[str]] = {}
    dependency_groups: dict[tuple[str, ...], list[str]] = {}
    for mapping in plan.mappings:
        mapping_shards: list[str] = []
        for source_name in mapping.source_names:
            shard_file = tensor_file_by_name.get(source_name)
            if shard_file is None:
                continue
            mapping_shards.append(shard_file)
            shard_sources = shard_to_sources.setdefault(shard_file, [])
            if source_name not in shard_sources:
                shard_sources.append(source_name)
            shard_targets = shard_to_targets.setdefault(shard_file, [])
            if mapping.target_name not in shard_targets:
                shard_targets.append(mapping.target_name)
        shard_key = tuple(sorted(set(mapping_shards)))
        if not shard_key:
            continue
        grouped_targets = dependency_groups.setdefault(shard_key, [])
        if mapping.target_name not in grouped_targets:
            grouped_targets.append(mapping.target_name)

    referenced_source_shards = tuple(
        file_name
        for file_name in inspection.shard_files
        if file_name in shard_to_sources
    )
    source_shard_dependencies = tuple(
        SourceShardDependency(
            shard_file=file_name,
            source_names=tuple(shard_to_sources[file_name]),
            target_names=tuple(shard_to_targets.get(file_name, ())),
        )
        for file_name in referenced_source_shards
    )
    mapping_dependency_groups = tuple(
        MappingDependencyGroup(
            shard_files=shard_files,
            target_names=tuple(target_names),
        )
        for shard_files, target_names in sorted(
            dependency_groups.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
    )
    return ExecutionDependencyPlan(
        referenced_source_tensors=referenced_source_tensors,
        referenced_source_shards=referenced_source_shards,
        source_shard_dependencies=source_shard_dependencies,
        mapping_dependency_groups=mapping_dependency_groups,
    )


def _load_source_weights(
    inspection: InspectionReport,
    dependencies: ExecutionDependencyPlan,
) -> tuple[dict[str, Any], str]:
    if not dependencies.referenced_source_shards:
        return {}, "selective_safetensors"

    try:
        from safetensors import safe_open
    except ImportError:
        # MLX exposes whole-file mx.load() but not name-filtered loading. If
        # safetensors selective reads are unavailable, keep the reduced logical
        # scope by loading only referenced shards and filtering tensors after
        # load. True streaming/incremental execution remains a later change.
        tensors: dict[str, Any] = {}
        referenced_names = set(dependencies.referenced_source_tensors)
        for file_name in dependencies.referenced_source_shards:
            file_path = inspection.resolved_path / file_name
            shard_tensors = mx.load(str(file_path))
            for name, value in shard_tensors.items():
                if name in referenced_names:
                    tensors[name] = value
        return tensors, "whole_shard_mx_load"

    tensors: dict[str, Any] = {}
    for group in dependencies.source_shard_dependencies:
        file_path = inspection.resolved_path / group.shard_file
        with safe_open(str(file_path), framework="np") as handle:
            for source_name in group.source_names:
                tensors[source_name] = handle.get_tensor(source_name)
    return tensors, "selective_safetensors"


@dataclass(frozen=True, slots=True)
class _ExecutionWriteOutcome:
    weight_files: tuple[str, ...]
    weight_index_file: str | None
    written_tensor_names: tuple[str, ...]
    loaded_source_tensors: tuple[str, ...]
    load_strategy: str
    materialization_strategy: str
    output_pack_groups: tuple[OutputPackGroup, ...]


class _SourceTensorStore:
    def __init__(
        self,
        inspection: InspectionReport,
        dependencies: ExecutionDependencyPlan,
        mappings: tuple[Any, ...],
    ) -> None:
        self._inspection = inspection
        self._dependencies = dependencies
        self._remaining_uses = Counter(
            source_name
            for mapping in mappings
            for source_name in mapping.source_names
        )
        self._tensor_file_by_name = {
            tensor.name: tensor.file
            for tensor in inspection.tensor_infos
            if tensor.name in dependencies.referenced_source_tensors
        }
        self._shard_source_names = {
            group.shard_file: set(group.source_names)
            for group in dependencies.source_shard_dependencies
        }
        self._loaded_tensors: dict[str, Any] = {}
        self._loaded_shards: dict[str, dict[str, Any]] = {}
        self._loaded_source_names: set[str] = set()
        self._safe_handles: dict[str, Any] = {}
        self._safe_open = None
        self._exit_stack = ExitStack()
        self.load_strategy = "whole_shard_mx_load"

    def __enter__(self) -> _SourceTensorStore:
        try:
            from safetensors import safe_open
        except ImportError:
            self._safe_open = None
            self.load_strategy = "whole_shard_mx_load"
            return self

        self._safe_open = safe_open
        for shard_file in self._dependencies.referenced_source_shards:
            file_path = self._inspection.resolved_path / shard_file
            self._safe_handles[shard_file] = self._exit_stack.enter_context(
                safe_open(str(file_path), framework="np")
            )
        self.load_strategy = "selective_safetensors"
        return self

    def __exit__(self, *_args: Any) -> None:
        self._exit_stack.close()

    @property
    def loaded_source_tensors(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaded_source_names))

    def get(self, source_name: str) -> Any:
        if source_name in self._loaded_tensors:
            return self._loaded_tensors[source_name]

        shard_file = self._tensor_file_by_name[source_name]
        if self._safe_open is None:
            shard = self._loaded_shards.get(shard_file)
            if shard is None:
                file_path = self._inspection.resolved_path / shard_file
                shard = mx.load(str(file_path))
                self._loaded_shards[shard_file] = shard
            value = shard[source_name]
        else:
            value = self._safe_handles[shard_file].get_tensor(source_name)

        self._loaded_tensors[source_name] = value
        self._loaded_source_names.add(source_name)
        return value

    def release(self, source_name: str) -> None:
        if source_name not in self._remaining_uses:
            return
        self._remaining_uses[source_name] -= 1
        if self._remaining_uses[source_name] > 0:
            return

        self._loaded_tensors.pop(source_name, None)
        shard_file = self._tensor_file_by_name.get(source_name)
        if shard_file is None or self._safe_open is not None:
            return
        if any(
            self._remaining_uses.get(name, 0) > 0
            for name in self._shard_source_names.get(shard_file, ())
        ):
            return
        self._loaded_shards.pop(shard_file, None)


def _execute_weight_write(
    inspection: InspectionReport,
    plan: ConversionPlan,
    destination: Path,
    *,
    dependencies: ExecutionDependencyPlan,
    max_shard_bytes: int | None,
) -> _ExecutionWriteOutcome:
    target_shapes = {entry.name: entry.shape for entry in plan.target_schema}
    ordered_mappings = _ordered_mappings(
        plan,
        sharded_output=max_shard_bytes is not None,
    )
    writer = _IncrementalWeightWriter(
        destination,
        max_shard_bytes=max_shard_bytes,
    )
    with _SourceTensorStore(inspection, dependencies, ordered_mappings) as source_store:
        for mapping in ordered_mappings:
            source_weights = {
                source_name: source_store.get(source_name)
                for source_name in mapping.source_names
            }
            materialized = _materialize_mapping(
                source_weights,
                mapping,
                expected_shape=target_shapes[mapping.target_name],
            )
            writer.add(mapping.target_name, materialized)
            for source_name in mapping.source_names:
                source_store.release(source_name)

        weight_files, index_file, written_names, output_pack_groups = writer.finalize()
        return _ExecutionWriteOutcome(
            weight_files=weight_files,
            weight_index_file=index_file,
            written_tensor_names=written_names,
            loaded_source_tensors=source_store.loaded_source_tensors,
            load_strategy=source_store.load_strategy,
            materialization_strategy=writer.materialization_strategy,
            output_pack_groups=output_pack_groups,
        )


def _ordered_mappings(
    plan: ConversionPlan,
    *,
    sharded_output: bool,
) -> tuple[Any, ...]:
    if not sharded_output:
        return tuple(plan.mappings)
    return tuple(sorted(plan.mappings, key=lambda mapping: mapping.target_name))


class _IncrementalWeightWriter:
    def __init__(
        self,
        destination: Path,
        *,
        max_shard_bytes: int | None,
    ) -> None:
        self._destination = destination
        self._max_shard_bytes = max_shard_bytes
        self._single_file_tensors: dict[str, Any] = {}
        self._current_shard: dict[str, Any] = {}
        self._current_names: list[str] = []
        self._current_size = 0
        self._written_names: list[str] = []
        self._total_size = 0
        self._staged_shards: list[tuple[Path, tuple[str, ...], int]] = []
        self.materialization_strategy = (
            "incremental_shard_buffered"
            if max_shard_bytes is not None
            else "single_file_buffered"
        )

    def add(self, target_name: str, tensor: Any) -> None:
        self._written_names.append(target_name)
        if self._max_shard_bytes is None:
            self._single_file_tensors[target_name] = tensor
            return

        tensor_size = int(getattr(tensor, "nbytes", 0))
        if self._current_shard and self._current_size + tensor_size > self._max_shard_bytes:
            self._flush_current_shard()
        self._current_shard[target_name] = tensor
        self._current_names.append(target_name)
        self._current_size += tensor_size
        self._total_size += tensor_size

    def finalize(
        self,
    ) -> tuple[tuple[str, ...], str | None, tuple[str, ...], tuple[OutputPackGroup, ...]]:
        # The current MLX writer is file-at-a-time, so single-file output still
        # requires buffering the final file payload. For sharded output we can
        # still flush bounded shard payloads earlier and keep only one shard
        # buffer live at a time.
        if self._max_shard_bytes is None or not self._written_names:
            file_name = "model.safetensors"
            mx.save_safetensors(
                str(self._destination / file_name),
                _to_mlx_tensors(self._single_file_tensors),
            )
            output_pack_groups = ()
            if self._single_file_tensors:
                output_pack_groups = (
                    OutputPackGroup(
                        target_names=tuple(self._single_file_tensors),
                        total_bytes=sum(
                            int(getattr(tensor, "nbytes", 0))
                            for tensor in self._single_file_tensors.values()
                        ),
                    ),
                )
            return (
                (file_name,),
                None,
                tuple(sorted(self._written_names)),
                output_pack_groups,
            )

        self._flush_current_shard()
        total = len(self._staged_shards)
        weight_files: list[str] = []
        weight_map: dict[str, str] = {}
        output_pack_groups: list[OutputPackGroup] = []
        for index, (temp_path, target_names, shard_size) in enumerate(
            self._staged_shards,
            start=1,
        ):
            file_name = f"model-{index:05d}-of-{total:05d}.safetensors"
            temp_path.rename(self._destination / file_name)
            weight_files.append(file_name)
            output_pack_groups.append(
                OutputPackGroup(
                    target_names=target_names,
                    total_bytes=shard_size,
                )
            )
            for target_name in target_names:
                weight_map[target_name] = file_name

        index_name = "model.safetensors.index.json"
        index_payload = {
            "metadata": {"total_size": self._total_size},
            "weight_map": weight_map,
        }
        (self._destination / index_name).write_text(
            json.dumps(index_payload, indent=2, sort_keys=True)
        )
        return (
            tuple(weight_files),
            index_name,
            tuple(sorted(self._written_names)),
            tuple(output_pack_groups),
        )

    def _flush_current_shard(self) -> None:
        if not self._current_shard:
            return
        temp_name = f".model-stage-{len(self._staged_shards) + 1:05d}.safetensors"
        temp_path = self._destination / temp_name
        mx.save_safetensors(str(temp_path), _to_mlx_tensors(self._current_shard))
        self._staged_shards.append(
            (
                temp_path,
                tuple(self._current_names),
                self._current_size,
            )
        )
        self._current_shard = {}
        self._current_names = []
        self._current_size = 0


def _materialize_target_tensors(
    source_weights: dict[str, Any],
    plan: ConversionPlan,
) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    target_shapes = {entry.name: entry.shape for entry in plan.target_schema}
    for mapping in plan.mappings:
        tensors[mapping.target_name] = _materialize_mapping(
            source_weights,
            mapping,
            expected_shape=target_shapes[mapping.target_name],
        )
    return tensors


def _materialize_mapping(
    source_weights: dict[str, Any],
    mapping: Any,
    *,
    expected_shape: tuple[int, ...],
) -> Any:
    materialized = _apply_mapping(source_weights, mapping)
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
    return materialized


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
    index_payload = {
        "metadata": {
            "total_size": sum(int(t.nbytes) for t in tensors.values()),
        },
        "weight_map": weight_map,
    }
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
