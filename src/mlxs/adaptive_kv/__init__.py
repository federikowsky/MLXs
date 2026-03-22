"""Adaptive KV subsystem exports."""

from mlxs.adaptive_kv.compatibility import CompatibilityResult, assess_generation_compatibility
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.exceptions import (
    AdaptiveKVCompatibilityError,
    AdaptiveKVError,
    AdaptiveKVRecoveryNotImplementedError,
    AdaptiveKVUnsupportedError,
)
from mlxs.adaptive_kv.manager import AdaptiveKVManager

__all__ = [
    "AdaptiveKVCompatibilityError",
    "AdaptiveKVConfig",
    "AdaptiveKVError",
    "AdaptiveKVManager",
    "AdaptiveKVRecoveryNotImplementedError",
    "AdaptiveKVUnsupportedError",
    "CompatibilityResult",
    "assess_generation_compatibility",
]
