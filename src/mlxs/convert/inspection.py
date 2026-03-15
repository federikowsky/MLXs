from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlxs.convert.errors import MissingRequiredConfigError, UnsupportedSourceFormatError
from mlxs.convert.types import ConversionOptions, InspectionReport, SourceKind, TensorInfo
from mlxs.convert.types import ConversionPhase

_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}

_MULTIMODAL_FILENAMES = {
    "feature_extractor_config.json",
    "image_processor_config.json",
    "preprocessor_config.json",
    "processor_config.json",
}


def inspect_source(
    source: str | Path,
    *,
    options: ConversionOptions | None = None,
) -> InspectionReport:
    opts = options or ConversionOptions()
    resolved_path, source_kind = _resolve_source_path(source, opts)
    config_path = resolved_path / "config.json"
    if not config_path.is_file():
        raise MissingRequiredConfigError(
            f"config.json not found in {resolved_path}",
            phase=ConversionPhase.INSPECTION,
        )

    config = json.loads(config_path.read_text())
    weight_files = sorted(resolved_path.glob("*.safetensors"))
    if not weight_files:
        _raise_unsupported_weight_layout(resolved_path)

    tensor_infos = _inspect_weight_files(weight_files)
    tokenizer_artifacts = _collect_artifacts(resolved_path, _TOKENIZER_FILENAMES)
    multimodal_artifacts = _collect_artifacts(resolved_path, _MULTIMODAL_FILENAMES)
    custom_code_indicators = _detect_custom_code_indicators(resolved_path, config)

    warnings: list[str] = []
    if custom_code_indicators and opts.fail_on_custom_code:
        warnings.append("custom_code_detected")

    return InspectionReport(
        source_kind=source_kind,
        source_id=str(source),
        resolved_path=resolved_path,
        config_path=config_path,
        config=config,
        weight_format="safetensors",
        sharded=len(weight_files) > 1,
        shard_files=tuple(path.name for path in weight_files),
        tensor_infos=tuple(tensor_infos),
        tokenizer_artifacts=tokenizer_artifacts,
        multimodal_artifacts=multimodal_artifacts,
        custom_code_indicators=tuple(custom_code_indicators),
        warnings=tuple(warnings),
    )


def _resolve_source_path(
    source: str | Path,
    options: ConversionOptions,
) -> tuple[Path, SourceKind]:
    path = Path(source)
    if path.is_dir():
        return path.resolve(), SourceKind.LOCAL

    from mlxs.load.resolve import resolve_model_path

    resolved = resolve_model_path(
        source,
        revision=options.revision,
        token=options.token,
    )
    return resolved.resolve(), SourceKind.HF_REPO


def _raise_unsupported_weight_layout(resolved_path: Path) -> None:
    if any(resolved_path.glob("*.bin")) or any(resolved_path.glob("*.pth")):
        raise UnsupportedSourceFormatError(
            f"Only safetensors sources are supported in V1: {resolved_path}",
            phase=ConversionPhase.INSPECTION,
        )
    raise UnsupportedSourceFormatError(
        f"No safetensors weight files found in {resolved_path}",
        phase=ConversionPhase.INSPECTION,
    )


def _inspect_weight_files(weight_files: list[Path]) -> list[TensorInfo]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise UnsupportedSourceFormatError(
            "safetensors is required to inspect source checkpoints",
            phase=ConversionPhase.INSPECTION,
        ) from exc

    tensor_infos: list[TensorInfo] = []
    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="np") as handle:
            for name in handle.keys():
                tensor_info = _tensor_info_from_handle(handle, name, weight_file.name)
                tensor_infos.append(tensor_info)
    return tensor_infos


def _tensor_info_from_handle(handle: Any, name: str, file_name: str) -> TensorInfo:
    slice_obj = handle.get_slice(name)
    dtype = getattr(slice_obj, "get_dtype", None)
    if callable(dtype):
        dtype_value = str(dtype())
    else:
        dtype_value = "unknown"
    shape = tuple(int(dim) for dim in slice_obj.get_shape())
    return TensorInfo(name=name, shape=shape, dtype=dtype_value, file=file_name)


def _collect_artifacts(resolved_path: Path, allowed_names: set[str]) -> tuple[str, ...]:
    artifacts: list[str] = []
    for entry in sorted(resolved_path.iterdir()):
        if not entry.is_file():
            continue
        if entry.name in allowed_names:
            artifacts.append(entry.name)
            continue
        if entry.name.endswith(".json") and (
            "tokenizer" in entry.name or "processor" in entry.name or "template" in entry.name
        ):
            artifacts.append(entry.name)
    return tuple(sorted(set(artifacts)))


def _detect_custom_code_indicators(resolved_path: Path, config: dict[str, Any]) -> list[str]:
    indicators: list[str] = []
    if "auto_map" in config:
        indicators.append("config:auto_map")
    if config.get("trust_remote_code"):
        indicators.append("config:trust_remote_code")
    for pattern in ("configuration_*.py", "modeling_*.py", "tokenization_*.py"):
        for entry in resolved_path.glob(pattern):
            indicators.append(f"file:{entry.name}")
    return sorted(set(indicators))
