"""Weight format registry — extensible dispatch for safetensors, PARO, AWQ, GPTQ (§7.2).

Detectors and loaders are registered per WeightFormat. Adding a new format
(e.g. AWQ, GPTQ) requires: (1) optional detector for auto, (2) loader
(path, model_config) -> (model, tokenizer). Only safetensors and PARO
are implemented; others raise a clear error.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import mlx.nn as nn

from mlxs._errors import ModelLoadError
from mlxs._types import WeightFormat
from mlxs.config.schema import ModelConfig
from mlxs.load.resolve import resolve_model_path
from mlxs.load.tokenizer import TokenizerWrapper

logger = logging.getLogger(__name__)

# Type for (path, model_config) -> (model, tokenizer). Model satisfies ModelProtocol.
LoaderFn = Callable[[Path, ModelConfig], tuple[nn.Module, TokenizerWrapper]]
DetectorFn = Callable[[Path], bool]

# Order for auto-detect: first match wins. Safetensors is fallback (no detector).
_AUTO_DETECT_ORDER = (WeightFormat.PARO, WeightFormat.AWQ, WeightFormat.GPTQ)

_FORMAT_DETECTORS: dict[WeightFormat, DetectorFn] = {}
_FORMAT_LOADERS: dict[WeightFormat, LoaderFn] = {}


def register_detector(format: WeightFormat) -> Callable[[DetectorFn], DetectorFn]:
    """Register a detector for auto-detect. Use as @register_detector(WeightFormat.PARO)."""

    def decorator(fn: DetectorFn) -> DetectorFn:
        _FORMAT_DETECTORS[format] = fn
        return fn

    return decorator


def register_loader(format: WeightFormat) -> Callable[[LoaderFn], LoaderFn]:
    """Register a loader for a weight format. Use as @register_loader(WeightFormat.PARO)."""

    def decorator(fn: LoaderFn) -> LoaderFn:
        _FORMAT_LOADERS[format] = fn
        return fn

    return decorator


def get_effective_format(path: Path, model_config: ModelConfig) -> WeightFormat:
    """Resolve effective weight format from config and optional auto-detection."""
    if model_config.weight_format != WeightFormat.AUTO:
        return model_config.weight_format
    for fmt in _AUTO_DETECT_ORDER:
        detector = _FORMAT_DETECTORS.get(fmt)
        if detector is not None and detector(path):
            logger.debug("Auto-detected weight format %s for %s", fmt.value, path)
            return fmt
    return WeightFormat.SAFETENSORS


def load_model_and_tokenizer(
    model_path: str | Path,
    model_config: ModelConfig,
) -> tuple[nn.Module, TokenizerWrapper]:
    """Load model and tokenizer according to effective weight format (§7.2).

    model_path can be a local directory or a Hugging Face model id; it is
    resolved once to a local path (HF ids are downloaded via snapshot_download).
    Optional model_config.model_hf_revision and model_config.model_hf_token
    control HF resolution. Dispatches to the registered loader for the format
    (safetensors, paro, etc.). For unimplemented formats (awq, gptq) raises
    ModelLoadError with a clear message.
    """
    path = resolve_model_path(
        model_path,
        revision=model_config.model_hf_revision,
        token=model_config.model_hf_token,
    )
    fmt = get_effective_format(path, model_config)
    loader = _FORMAT_LOADERS.get(fmt)
    if loader is None:
        raise ModelLoadError(
            f"Weight format '{fmt.value}' is not implemented. "
            "Use weight_format=safetensors or paro, or install mlxs[paro] for PARO."
        )
    return loader(path, model_config)


# --- Safetensors (default path) ---


@register_loader(WeightFormat.SAFETENSORS)
def _load_safetensors(path: Path, model_config: ModelConfig) -> tuple[nn.Module, TokenizerWrapper]:
    """Native safetensors + optional nn.quantize from config.json."""
    from mlxs.load.loader import load_model, load_tokenizer

    lazy = model_config.lazy_load and not model_config.preload
    model = load_model(path, lazy=lazy)
    tokenizer = load_tokenizer(path, trust_remote_code=model_config.trust_remote_code)
    return model, tokenizer


# --- PARO ---


@register_detector(WeightFormat.PARO)
def _detect_paro(path: Path) -> bool:
    """Delegate to paro module (quant_method == paroquant)."""
    from mlxs.load.paro import _detect_paro as detect
    return detect(path)


@register_loader(WeightFormat.PARO)
def _load_paro(path: Path, model_config: ModelConfig) -> tuple[nn.Module, TokenizerWrapper]:
    """Load PARO model and tokenizer via paroquant MLX backend."""
    from mlxs.load.paro import load_paro_model_and_tokenizer

    lazy = model_config.lazy_load and not model_config.preload
    return load_paro_model_and_tokenizer(path, model_config, lazy=lazy)


# --- AWQ / GPTQ (placeholders for future) ---


def _detect_quant_method(path: Path, quant_method: str) -> bool:
    """Return True if config.json has quantization_config.quant_method == quant_method."""
    from mlxs.load.weights import load_config
    try:
        config = load_config(path)
        return config.get("quantization_config", {}).get("quant_method") == quant_method
    except FileNotFoundError:
        return False


@register_detector(WeightFormat.AWQ)
def _detect_awq(path: Path) -> bool:
    """Detect AWQ format from config (quant_method == awq)."""
    return _detect_quant_method(path, "awq")


@register_loader(WeightFormat.AWQ)
def _load_awq(_path: Path, _model_config: ModelConfig) -> tuple[nn.Module, TokenizerWrapper]:
    """Placeholder loader; raises ModelLoadError (AWQ not implemented)."""
    raise ModelLoadError(
        "AWQ weight format is not implemented yet. "
        "Use weight_format=safetensors or paro, or contribute an AWQ loader."
    )


@register_detector(WeightFormat.GPTQ)
def _detect_gptq(path: Path) -> bool:
    """Detect GPTQ format from config (quant_method == gptq)."""
    return _detect_quant_method(path, "gptq")


@register_loader(WeightFormat.GPTQ)
def _load_gptq(_path: Path, _model_config: ModelConfig) -> tuple[nn.Module, TokenizerWrapper]:
    """Placeholder loader; raises ModelLoadError (GPTQ not implemented)."""
    raise ModelLoadError(
        "GPTQ weight format is not implemented yet. "
        "Use weight_format=safetensors or paro, or contribute a GPTQ loader."
    )
