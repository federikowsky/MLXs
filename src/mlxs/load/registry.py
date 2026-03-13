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
    "afm7": ("mlxs.models.afm7", "Model", "ModelArgs"),
    "afmoe": ("mlxs.models.afmoe", "Model", "ModelArgs"),
    "apertus": ("mlxs.models.apertus", "Model", "ModelArgs"),
    "baichuan_m1": ("mlxs.models.baichuan_m1", "Model", "ModelArgs"),
    "bitnet": ("mlxs.models.bitnet", "Model", "ModelArgs"),
    "bailing_moe_linear": ("mlxs.models.bailing_moe_linear", "Model", "ModelArgs"),
    "cohere": ("mlxs.models.cohere", "Model", "ModelArgs"),
    "cohere2": ("mlxs.models.cohere2", "Model", "ModelArgs"),
    "dbrx": ("mlxs.models.dbrx", "Model", "ModelArgs"),
    "deepseek": ("mlxs.models.deepseek", "Model", "ModelArgs"),
    "falcon_h1": ("mlxs.models.falcon_h1", "Model", "ModelArgs"),
    "deepseek_v2": ("mlxs.models.deepseek_v2", "Model", "ModelArgs"),
    "deepseek_v3": ("mlxs.models.deepseek_v3", "Model", "ModelArgs"),
    "deepseek_v32": ("mlxs.models.deepseek_v32", "Model", "ModelArgs"),
    "dots1": ("mlxs.models.dots1", "Model", "ModelArgs"),
    "gemma": ("mlxs.models.gemma", "Model", "ModelArgs"),
    "gemma2": ("mlxs.models.gemma2", "Model", "ModelArgs"),
    "gemma3": ("mlxs.models.gemma3", "Model", "ModelArgs"),
    "gemma3n": ("mlxs.models.gemma3n", "Model", "ModelArgs"),
    "glm": ("mlxs.models.glm", "Model", "ModelArgs"),
    "glm4": ("mlxs.models.glm4", "Model", "ModelArgs"),
    "glm4_moe": ("mlxs.models.glm4_moe", "Model", "ModelArgs"),
    "glm4_moe_lite": ("mlxs.models.glm4_moe_lite", "Model", "ModelArgs"),
    "glm_moe_dsa": ("mlxs.models.glm_moe_dsa", "Model", "ModelArgs"),
    "granite": ("mlxs.models.granite", "Model", "ModelArgs"),
    "lille_130m": ("mlxs.models.lille_130m", "Model", "ModelArgs"),
    "granitemoe": ("mlxs.models.granitemoe", "Model", "ModelArgs"),
    "granitemoehybrid": ("mlxs.models.granitemoehybrid", "Model", "ModelArgs"),
    "gpt2": ("mlxs.models.gpt2", "Model", "ModelArgs"),
    "hunyuan": ("mlxs.models.hunyuan", "Model", "ModelArgs"),
    "iquestloopcoder": ("mlxs.models.iquestloopcoder", "Model", "ModelArgs"),
    "hunyuan_v1_dense": ("mlxs.models.hunyuan_v1_dense", "Model", "ModelArgs"),
    "gpt_bigcode": ("mlxs.models.gpt_bigcode", "Model", "ModelArgs"),
    "gpt_neox": ("mlxs.models.gpt_neox", "Model", "ModelArgs"),
    "gpt_oss": ("mlxs.models.gpt_oss", "Model", "ModelArgs"),
    "internlm2": ("mlxs.models.internlm2", "Model", "ModelArgs"),
    "internlm3": ("mlxs.models.internlm3", "Model", "ModelArgs"),
    "jamba": ("mlxs.models.jamba", "Model", "ModelArgs"),
    "lfm2": ("mlxs.models.lfm2", "Model", "ModelArgs"),
    "lfm2_moe": ("mlxs.models.lfm2_moe", "Model", "ModelArgs"),
    "lfm2_vl": ("mlxs.models.lfm2_vl", "Model", "ModelArgs"),
    "klear": ("mlxs.models.klear", "Model", "ModelArgs"),
    "kimi_k25": ("mlxs.models.kimi_k25", "Model", "ModelArgs"),
    "kimi_linear": ("mlxs.models.kimi_linear", "Model", "ModelArgs"),
    "kimi_vl": ("mlxs.models.kimi_vl", "Model", "ModelArgs"),
    "longcat_flash": ("mlxs.models.longcat_flash", "Model", "ModelArgs"),
    "longcat_flash_ngram": ("mlxs.models.longcat_flash_ngram", "Model", "ModelArgs"),
    "llama": ("mlxs.models.llama", "Model", "ModelArgs"),
    "llama4": ("mlxs.models.llama4", "Model", "ModelArgs"),
    "mamba": ("mlxs.models.mamba", "Model", "ModelArgs"),
    "mamba2": ("mlxs.models.mamba2", "Model", "ModelArgs"),
    "minicpm": ("mlxs.models.minicpm", "Model", "ModelArgs"),
    "minicpm3": ("mlxs.models.minicpm3", "Model", "ModelArgs"),
    "mimo": ("mlxs.models.mimo", "Model", "ModelArgs"),
    "minimax": ("mlxs.models.minimax", "Model", "ModelArgs"),
    "mimo_v2_flash": ("mlxs.models.mimo_v2_flash", "Model", "ModelArgs"),
    "nanochat": ("mlxs.models.nanochat", "Model", "ModelArgs"),
    "ministral3": ("mlxs.models.ministral3", "Model", "ModelArgs"),
    "mistral3": ("mlxs.models.mistral3", "Model", "ModelArgs"),
    "mixtral": ("mlxs.models.mixtral", "Model", "ModelArgs"),
    "olmo": ("mlxs.models.olmo", "Model", "ModelArgs"),
    "olmo2": ("mlxs.models.olmo2", "Model", "ModelArgs"),
    "olmo3": ("mlxs.models.olmo3", "Model", "ModelArgs"),
    "olmoe": ("mlxs.models.olmoe", "Model", "ModelArgs"),
    "openelm": ("mlxs.models.openelm", "Model", "ModelArgs"),
    "ernie4_5_moe": ("mlxs.models.ernie4_5_moe", "Model", "ModelArgs"),
    "ernie4_5": ("mlxs.models.ernie4_5", "Model", "ModelArgs"),
    "exaone": ("mlxs.models.exaone", "Model", "ModelArgs"),
    "exaone4": ("mlxs.models.exaone4", "Model", "ModelArgs"),
    "exaone_moe": ("mlxs.models.exaone_moe", "Model", "ModelArgs"),
    "nemotron": ("mlxs.models.nemotron", "Model", "ModelArgs"),
    "nemotron_h": ("mlxs.models.nemotron_h", "Model", "ModelArgs"),
    "nemotron_nas": ("mlxs.models.nemotron_nas", "Model", "ModelArgs"),
    "plamo": ("mlxs.models.plamo", "Model", "ModelArgs"),
    "plamo2": ("mlxs.models.plamo2", "Model", "ModelArgs"),
    "helium": ("mlxs.models.helium", "Model", "ModelArgs"),
    "phi": ("mlxs.models.phi", "Model", "ModelArgs"),
    "phi3": ("mlxs.models.phi3", "Model", "ModelArgs"),
    "phi3small": ("mlxs.models.phi3small", "Model", "ModelArgs"),
    "phixtral": ("mlxs.models.phixtral", "Model", "ModelArgs"),
    "phimoe": ("mlxs.models.phimoe", "Model", "ModelArgs"),
    "pixtral": ("mlxs.models.pixtral", "Model", "ModelArgs"),
    "qwen2": ("mlxs.models.qwen", "Model", "ModelArgs"),
    "qwen2_moe": ("mlxs.models.qwen2_moe", "Model", "ModelArgs"),
    "qwen2_vl": ("mlxs.models.qwen2_vl", "Model", "ModelArgs"),
    "qwen3": ("mlxs.models.qwen3", "Model", "ModelArgs"),
    "qwen3_5": ("mlxs.models.qwen3_5", "Model", "ModelArgs"),
    "qwen3_5_moe": ("mlxs.models.qwen3_5_moe", "Model", "ModelArgs"),
    "qwen3_moe": ("mlxs.models.qwen3_moe", "Model", "ModelArgs"),
    "qwen3_vl_moe": ("mlxs.models.qwen3_vl_moe", "Model", "ModelArgs"),
    "qwen3_next": ("mlxs.models.qwen3_next", "Model", "ModelArgs"),
    "qwen3_vl": ("mlxs.models.qwen3_vl", "Model", "ModelArgs"),
    "recurrent_gemma": ("mlxs.models.recurrent_gemma", "Model", "ModelArgs"),
    "rwkv7": ("mlxs.models.rwkv7", "Model", "ModelArgs"),
    "solar_open": ("mlxs.models.solar_open", "Model", "ModelArgs"),
    "seed_oss": ("mlxs.models.seed_oss", "Model", "ModelArgs"),
    "smollm3": ("mlxs.models.smollm3", "Model", "ModelArgs"),
    "stablelm": ("mlxs.models.stablelm", "Model", "ModelArgs"),
    "starcoder2": ("mlxs.models.starcoder2", "Model", "ModelArgs"),
    "step3p5": ("mlxs.models.step3p5", "Model", "ModelArgs"),
    "bailing_moe": ("mlxs.models.bailing_moe", "Model", "ModelArgs"),
    "telechat3": ("mlxs.models.telechat3", "Model", "ModelArgs"),
    "youtu_llm": ("mlxs.models.youtu_llm", "Model", "ModelArgs"),
}

# Aliases for config.json model_type that map to existing architectures (mlx_lm-compatible)
_MODEL_REMAPPING: dict[str, str] = {
    "falcon_mamba": "mamba",
    "gemma3_text": "gemma3",
    "griffin": "recurrent_gemma",
    "iquestcoder": "llama",
    "joyai_llm_flash": "deepseek_v3",
    "llama4_text": "llama4",
    "kimi_k2": "deepseek_v3",
    "mistral": "llama",
    "nemotron-nas": "nemotron_nas",
    "phi3_small": "phi3small",
    "qwen2_5_vl": "qwen2_vl",
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
