"""Shared SigLIP construction helpers for wrapper-style multimodal families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mlxs.models.vision.siglip import SigLIPVisionModel, VisionConfig

_SIGLIP_CONFIG_FIELDS = frozenset(VisionConfig.__dataclass_fields__)


def normalize_siglip_vision_config(raw_config: Mapping[str, Any]) -> VisionConfig:
    """Normalize HF-like raw config mappings into the repo's VisionConfig.

    Supported aliases are limited to the shapes already observed in P0:
    - `hidden_size` as fallback source for `embed_dim`
    - `out_hidden_size` as LM-facing merged hidden size
    - `in_chans` as alias for `in_channels`
    """

    mapped = dict(raw_config)
    mapped["embed_dim"] = raw_config.get("embed_dim", raw_config.get("hidden_size"))
    mapped["hidden_size"] = raw_config.get(
        "out_hidden_size",
        raw_config.get("hidden_size", VisionConfig.hidden_size),
    )
    mapped["in_channels"] = raw_config.get(
        "in_channels",
        raw_config.get("in_chans", VisionConfig.in_channels),
    )
    filtered = {
        key: value
        for key, value in mapped.items()
        if key in _SIGLIP_CONFIG_FIELDS
    }
    return VisionConfig(**filtered)


def build_siglip_vision_tower(raw_config: Mapping[str, Any]) -> SigLIPVisionModel:
    """Build a SigLIP vision tower from a raw config mapping."""

    return SigLIPVisionModel(normalize_siglip_vision_config(raw_config))
