"""Model registry — model_type to (Model, ModelArgs) mapping (§7.2, AC17).

Adding a new architecture requires only:
1. A new file in models/ with Model and ModelArgs classes
2. An entry in MODEL_REGISTRY below

No changes to generate, cache, batch, or server.
"""

from __future__ import annotations

from typing import Any

# Registry: model_type string → (module_path, Model class name, ModelArgs class name)
# Lazy imports to avoid loading all model code at startup.
MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "llama": ("mlxs.models.llama", "Model", "ModelArgs"),
    "qwen2": ("mlxs.models.qwen", "Model", "ModelArgs"),
}

# Aliases for common model types that use existing architectures
_MODEL_REMAPPING: dict[str, str] = {
    "mistral": "llama",
    "qwen3": "qwen2",  # Qwen3 dense uses same architecture as Qwen2
}


def get_model_classes(model_type: str) -> tuple[type[Any], type[Any]]:
    """Look up Model and ModelArgs classes for a model_type.

    Args:
        model_type: The ``model_type`` field from config.json.

    Returns:
        Tuple of (ModelClass, ModelArgsClass).

    Raises:
        ValueError: If model_type is not in the registry.
    """
    import importlib

    canonical = _MODEL_REMAPPING.get(model_type, model_type)
    entry = MODEL_REGISTRY.get(canonical)
    if entry is None:
        supported = sorted({*MODEL_REGISTRY, *_MODEL_REMAPPING})
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported: {', '.join(supported)}"
        )

    module_path, model_cls_name, args_cls_name = entry
    module = importlib.import_module(module_path)
    return getattr(module, model_cls_name), getattr(module, args_cls_name)
