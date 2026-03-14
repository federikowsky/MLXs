"""High-level model and tokenizer loading (FR1, FR11, AC10).

Entry points for loading a complete model+tokenizer. model_path can be a local
directory or a Hugging Face model id (resolved automatically). Supports multiple
weight formats (safetensors, PARO, etc.) via formats registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlxs._errors import ModelLoadError
from mlxs.load.registry import get_model_classes
from mlxs.load.tokenizer import TokenizerWrapper, load_hf_tokenizer
from mlxs.load.weights import load_config, load_weights

if TYPE_CHECKING:
    from mlxs.config.schema import ModelConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str | Path,
    model_config: ModelConfig,
) -> tuple[nn.Module, TokenizerWrapper]:
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
) -> nn.Module:
    """Load a model from a local path.

    Reads config.json, resolves the architecture via the registry,
    instantiates the model, and loads safetensors weights.

    Args:
        model_path: Local directory containing config.json + *.safetensors.
        lazy: If True, defer weight loading (FR11). Weights load on first forward.

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

    args = ModelArgsClass.from_dict(config)
    model = ModelClass(args)

    if lazy:
        logger.info("Lazy load enabled — weights deferred until first call")
        return model

    try:
        load_weights(path, model)
    except FileNotFoundError as exc:
        raise ModelLoadError(str(exc)) from exc

    mx.eval(model.parameters())
    logger.info("Model loaded: %s (%s) from %s", model_type, ModelClass.__name__, path)
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
