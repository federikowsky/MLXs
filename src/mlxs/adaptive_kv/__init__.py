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
from mlxs.adaptive_kv.runtime import (
    AdapterCapabilities,
    AdaptiveKVCapabilityProvider,
    AdaptiveKVReplayBackend,
    AdaptiveKVRuntimeAdapter,
    AdaptiveKVRuntimeSubstrate,
    CapabilityStatus,
    ComposedAdaptiveKVRuntimeAdapter,
    RuntimeFamily,
    RuntimeFamilyBindings,
    RuntimeFamilyDescriptor,
    SupportLevel,
)

__all__ = [
    "AdapterCapabilities",
    "AdaptiveKVCapabilityProvider",
    "AdaptiveKVCompatibilityError",
    "AdaptiveKVConfig",
    "AdaptiveKVError",
    "AdaptiveKVManager",
    "AdaptiveKVRecoveryNotImplementedError",
    "AdaptiveKVReplayBackend",
    "AdaptiveKVRuntimeAdapter",
    "AdaptiveKVRuntimeSubstrate",
    "AdaptiveKVUnsupportedError",
    "CapabilityStatus",
    "CompatibilityResult",
    "ComposedAdaptiveKVRuntimeAdapter",
    "RuntimeFamily",
    "RuntimeFamilyBindings",
    "RuntimeFamilyDescriptor",
    "SupportLevel",
    "assess_generation_compatibility",
    "select_generation_adapter",
]
