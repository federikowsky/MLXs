"""Inter-module contracts for MLXs (O8, §2.4, §9).

All protocols are re-exported from this package for convenience:

    from mlxs.protocols import ModelProtocol, CacheProtocol, ...

No implementations live here — only structural typing contracts.
"""

from mlxs.protocols.batch import BatchSchedulerProtocol
from mlxs.protocols.cache import CacheProtocol
from mlxs.protocols.generate import GenerateProtocol
from mlxs.protocols.metrics import MetricsProtocol
from mlxs.protocols.model import ModelProtocol
from mlxs.protocols.multimodal import MultimodalModelProtocol
from mlxs.protocols.prompt_cache import PromptCacheProtocol

__all__ = [
    "BatchSchedulerProtocol",
    "CacheProtocol",
    "GenerateProtocol",
    "MetricsProtocol",
    "ModelProtocol",
    "MultimodalModelProtocol",
    "PromptCacheProtocol",
]
