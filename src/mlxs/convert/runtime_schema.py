from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from mlxs._types import ModelMode
from mlxs.convert.types import RuntimeTensorSchemaEntry
from mlxs.load.registry import get_model_classes, instantiate_model

_DTYPE_ALIASES = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "bool": "BOOL",
    "bool_": "BOOL",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
    "f16": "F16",
    "f32": "F32",
    "f64": "F64",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "i16": "I16",
    "i32": "I32",
    "i64": "I64",
    "uint16": "U16",
    "uint32": "U32",
    "uint64": "U64",
    "u16": "U16",
    "u32": "U32",
    "u64": "U64",
}


def export_runtime_schema(
    config: dict[str, Any],
    model_type: str,
    *,
    model_mode: str | ModelMode,
) -> tuple[RuntimeTensorSchemaEntry, ...]:
    _, ModelArgsClass = get_model_classes(model_type)
    args = ModelArgsClass.from_dict(config)
    model = instantiate_model(model_type, args, model_mode=_coerce_model_mode(model_mode))
    flat = _flatten_parameter_tree(model.parameters())
    return tuple(
        RuntimeTensorSchemaEntry(
            name=name,
            shape=tuple(int(dim) for dim in value.shape),
            dtype=_normalized_dtype_name(getattr(value, "dtype", None)),
        )
        for name, value in sorted(flat.items())
    )


def runtime_schema_hash(entries: Iterable[RuntimeTensorSchemaEntry]) -> str:
    payload = [
        {
            "name": entry.name,
            "shape": [int(dim) for dim in entry.shape],
            "dtype": _normalized_dtype_name(entry.dtype),
        }
        for entry in entries
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coerce_model_mode(model_mode: str | ModelMode) -> ModelMode:
    if isinstance(model_mode, ModelMode):
        return model_mode
    return ModelMode(str(model_mode))


def _flatten_parameter_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    if hasattr(tree, "shape"):
        return {prefix: tree}
    if isinstance(tree, dict):
        flat: dict[str, Any] = {}
        for key, value in tree.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_parameter_tree(value, child_prefix))
        return flat
    if isinstance(tree, (list, tuple)):
        flat = {}
        for index, value in enumerate(tree):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flat.update(_flatten_parameter_tree(value, child_prefix))
        return flat
    return {}


def _normalized_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    raw = str(dtype)
    key = raw.split(".")[-1].lower()
    return _DTYPE_ALIASES.get(raw.lower(), _DTYPE_ALIASES.get(key, raw))
