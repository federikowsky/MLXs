"""Adaptive KV subsystem exports."""

from mlxs.adaptive_kv.compatibility import (
    CompatibilityResult,
    assess_generation_compatibility,
    select_generation_adapter,
)
from mlxs.adaptive_kv.config import AdaptiveKVConfig
from mlxs.adaptive_kv.exceptions import (
    AdaptiveKVCompatibilityError,
    AdaptiveKVError,
    AdaptiveKVRecoveryNotImplementedError,
    AdaptiveKVUnsupportedError,
)
from mlxs.adaptive_kv.manager import AdaptiveKVManager
from mlxs.adaptive_kv.runtime import AdapterCapabilities, CapabilityStatus, SupportLevel

__all__ = [
    "AdaptiveKVCompatibilityError",
    "AdaptiveKVConfig",
    "AdapterCapabilities",
    "AdaptiveKVError",
    "AdaptiveKVManager",
    "AdaptiveKVRecoveryNotImplementedError",
    "AdaptiveKVUnsupportedError",
    "CapabilityStatus",
    "CompatibilityResult",
    "assess_generation_compatibility",
    "select_generation_adapter",
    "SupportLevel",
]
