"""High-level model and tokenizer loading (FR1, FR11, AC10).

Entry points for loading a complete model+tokenizer. model_path can be a local
directory or a Hugging Face model id (resolved automatically). Supports multiple
weight formats (safetensors, PARO, etc.) via formats registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from mlxs._errors import ModelLoadError
from mlxs._types import ModelMode
from mlxs.load.registry import get_model_classes
from mlxs.load.tokenizer import TokenizerWrapper, load_hf_tokenizer
from mlxs.load.weights import load_config, load_weights

if TYPE_CHECKING:
    from mlxs.config.schema import ModelConfig
    ModuleT = Any
else:
    ModuleT = nn.Module

logger = logging.getLogger(__name__)

_DEFERRED_MULTIMODAL_MODEL_TYPES = {"qwen3_5_moe"}


def load_model_and_tokenizer(
    model_path: str | Path,
    model_config: ModelConfig,
) -> tuple[ModuleT, TokenizerWrapper]:
    """Load model and tokenizer according to config weight_format (FR1, §7.2).

    model_path can be a local directory or a Hugging Face model id; it is
    resolved once before format detection. Optional model_config.model_hf_revision
    and model_config.model_hf_token control HF resolution. Dispatches to the
    appropriate loader (safetensors, paro, etc.). Use from the composition root
    when format-aware loading is desired.
    """
    from mlxs.load.formats import load_model_and_tokenizer as _load_by_format
    return _load_by_format(model_path, model_config)


def load_model(
    model_path: str | Path,
    *,
    lazy: bool = False,
    model_mode: ModelMode = ModelMode.AUTO,
) -> ModuleT:
    """Load a model from a local path.

    Reads config.json, resolves the architecture via the registry,
    instantiates the model, and loads safetensors weights.

    Args:
        model_path: Local directory containing config.json + *.safetensors.
        lazy: If True, defer weight loading (FR11). Weights load on first forward.
        model_mode: TEXT, MULTIMODAL, or AUTO. AUTO resolves to MULTIMODAL
            if config.json contains vision_config, otherwise TEXT (§7.4).

    Returns:
        Model instance (satisfies ModelProtocol).

    Raises:
        ModelLoadError: On missing files, unsupported architecture, or load failure.
    """
    path = Path(model_path)

    try:
        config = load_config(path)
    except FileNotFoundError as exc:
        raise ModelLoadError(str(exc)) from exc

    model_type = config.get("model_type")
    if model_type is None:
        raise ModelLoadError(f"config.json in {path} missing 'model_type' field")

    try:
        ModelClass, ModelArgsClass = get_model_classes(model_type)
    except ValueError as exc:
        raise ModelLoadError(str(exc)) from exc

    # Resolve AUTO mode (§7.4)
    resolved_mode = model_mode
    if model_mode == ModelMode.AUTO:
        has_vision = "vision_config" in config or "visual_config" in config
        resolved_mode = ModelMode.MULTIMODAL if has_vision else ModelMode.TEXT

    # Validate MULTIMODAL requires vision_config
    if resolved_mode == ModelMode.MULTIMODAL:
        if "vision_config" not in config and "visual_config" not in config:
            raise ModelLoadError(
                f"model_mode=multimodal requested but {model_type} config.json "
                f"has no vision_config. This model does not support vision."
            )
        if model_type in _DEFERRED_MULTIMODAL_MODEL_TYPES:
            raise ModelLoadError(
                f"{model_type} exposes multimodal-compatible config fields, "
                "but multimodal runtime support is intentionally deferred."
            )

    args = ModelArgsClass.from_dict(config)

    # Pass model_mode if the constructor accepts it
    import inspect

    sig = inspect.signature(ModelClass.__init__)
    if "model_mode" in sig.parameters:
        model = ModelClass(args, model_mode=resolved_mode)
    else:
        model = ModelClass(args)

    if lazy:
        logger.info("Lazy load enabled — weights deferred until first call")
        return model

    try:
        load_weights(path, model)
    except FileNotFoundError as exc:
        raise ModelLoadError(str(exc)) from exc

    mx.eval(model.parameters())
    logger.info(
        "Model loaded: %s (%s) mode=%s from %s",
        model_type, ModelClass.__name__, resolved_mode.value, path,
    )
    return model


def load_tokenizer(
    model_path: str | Path,
    *,
    trust_remote_code: bool = False,
) -> TokenizerWrapper:
    """Load a tokenizer from a local model path.

    Args:
        model_path: Local directory or HF repo id.
        trust_remote_code: Allow custom tokenizer code.

    Returns:
        TokenizerWrapper instance.

    Raises:
        ModelLoadError: On tokenizer load failure.
    """
    try:
        return load_hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load tokenizer from {model_path}: {exc}") from exc
