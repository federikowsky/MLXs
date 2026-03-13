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
    "cohere": ("mlxs.models.cohere", "Model", "ModelArgs"),
    "cohere2": ("mlxs.models.cohere2", "Model", "ModelArgs"),
    "dbrx": ("mlxs.models.dbrx", "Model", "ModelArgs"),
    "deepseek": ("mlxs.models.deepseek", "Model", "ModelArgs"),
    "falcon_h1": ("mlxs.models.falcon_h1", "Model", "ModelArgs"),
    "deepseek_v2": ("mlxs.models.deepseek_v2", "Model", "ModelArgs"),
    "deepseek_v3": ("mlxs.models.deepseek_v3", "Model", "ModelArgs"),
    "gemma": ("mlxs.models.gemma", "Model", "ModelArgs"),
    "gemma2": ("mlxs.models.gemma2", "Model", "ModelArgs"),
    "granite": ("mlxs.models.granite", "Model", "ModelArgs"),
    "granitemoe": ("mlxs.models.granitemoe", "Model", "ModelArgs"),
    "gpt2": ("mlxs.models.gpt2", "Model", "ModelArgs"),
    "gpt_bigcode": ("mlxs.models.gpt_bigcode", "Model", "ModelArgs"),
    "gpt_neox": ("mlxs.models.gpt_neox", "Model", "ModelArgs"),
    "internlm2": ("mlxs.models.internlm2", "Model", "ModelArgs"),
    "llama": ("mlxs.models.llama", "Model", "ModelArgs"),
    "mamba": ("mlxs.models.mamba", "Model", "ModelArgs"),
    "mamba2": ("mlxs.models.mamba2", "Model", "ModelArgs"),
    "minicpm": ("mlxs.models.minicpm", "Model", "ModelArgs"),
    "mixtral": ("mlxs.models.mixtral", "Model", "ModelArgs"),
    "olmoe": ("mlxs.models.olmoe", "Model", "ModelArgs"),
    "openelm": ("mlxs.models.openelm", "Model", "ModelArgs"),
    "phi": ("mlxs.models.phi", "Model", "ModelArgs"),
    "phi3": ("mlxs.models.phi3", "Model", "ModelArgs"),
    "phimoe": ("mlxs.models.phimoe", "Model", "ModelArgs"),
    "qwen2": ("mlxs.models.qwen", "Model", "ModelArgs"),
    "qwen2_moe": ("mlxs.models.qwen2_moe", "Model", "ModelArgs"),
    "qwen3": ("mlxs.models.qwen3", "Model", "ModelArgs"),
    "qwen3_moe": ("mlxs.models.qwen3_moe", "Model", "ModelArgs"),
    "stablelm": ("mlxs.models.stablelm", "Model", "ModelArgs"),
    "starcoder2": ("mlxs.models.starcoder2", "Model", "ModelArgs"),
}

# Aliases for config.json model_type that map to existing architectures (mlx_lm-compatible)
_MODEL_REMAPPING: dict[str, str] = {
    "falcon_mamba": "mamba",
    "iquestcoder": "llama",
    "joyai_llm_flash": "deepseek_v3",
    "kimi_k2": "deepseek_v3",
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
