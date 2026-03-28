"""Adaptive KV subsystem exports."""

from mlxs.adaptive_kv.block_types import ResidentProfile
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
from mlxs.adaptive_kv.resident import (
    ExecutionFragment,
    ExecutionSlab,
    ResidentBackend,
    ResidentBlockHandle,
    ResidentExecutionFabric,
    ResidentExecutionMode,
    ResidentProfileDescriptor,
    SlabBlockRange,
    TurboQuantResidentBackend,
)
from mlxs.adaptive_kv.runtime import (
    AdapterCapabilities,
    AdaptiveKVCapabilityProvider,
    AdaptiveKVReplayBackend,
    AdaptiveKVRuntimeAdapter,
    AdaptiveKVRuntimeSubstrate,
    CapabilityStatus,
    ComposedAdaptiveKVRuntimeAdapter,
    FamilyProfileCapability,
    RuntimeFamily,
    RuntimeFamilyBindings,
    RuntimeFamilyDescriptor,
    SupportLevel,
)
from mlxs.adaptive_kv.storage import ExecutionSliceRef, ResidentStateView

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
    "ExecutionFragment",
    "ExecutionSlab",
    "ExecutionSliceRef",
    "FamilyProfileCapability",
    "ResidentBackend",
    "ResidentBlockHandle",
    "ResidentExecutionFabric",
    "ResidentExecutionMode",
    "ResidentProfile",
    "ResidentProfileDescriptor",
    "ResidentStateView",
    "RuntimeFamily",
    "RuntimeFamilyBindings",
    "RuntimeFamilyDescriptor",
    "SlabBlockRange",
    "SupportLevel",
    "TurboQuantResidentBackend",
    "assess_generation_compatibility",
    "select_generation_adapter",
]
