"""Safetensors weight loading — single and sharded (FR1, §7.2).

Compatible with mlx_lm-converted models (same layout).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def load_config(model_path: Path) -> dict[str, Any]:
    """Load model config.json from a local path."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with config_path.open() as f:
        return json.load(f)


def load_weights(model_path: Path, model: nn.Module) -> None:
    """Load safetensors weights into a model.

    Supports single-file and sharded (*.safetensors) layouts.
    Calls model.sanitize() if available to remap weight keys.

    Args:
        model_path: Directory containing .safetensors file(s).
        model: Model instance to load weights into.
    """
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)))

    # Allow model to remap keys (e.g. remove rotary_emb.inv_freq)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Apply quantization config if present (skip for Ouro: pre-quantized weights
    # are dequantized in model.sanitize and loaded as full precision)
    config = load_config(model_path)
    if "quantization" in config and config.get("model_type") != "ouro":
        q = config["quantization"]
        nn.quantize(
            model,
            group_size=q.get("group_size", 64),
            bits=q.get("bits", 4),
        )

    model.load_weights(list(weights.items()), strict=False)
    logger.info("Loaded %d weight tensors from %d file(s)", len(weights), len(weight_files))
