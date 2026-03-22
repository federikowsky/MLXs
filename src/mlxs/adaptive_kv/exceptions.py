"""Adaptive KV exceptions."""

from __future__ import annotations


class AdaptiveKVError(RuntimeError):
    """Base class for adaptive KV failures."""


class AdaptiveKVCompatibilityError(AdaptiveKVError):
    """Raised when adaptive mode is incompatible with the current request."""


class AdaptiveKVUnsupportedError(AdaptiveKVError):
    """Raised when adaptive mode is not supported for the current runtime."""


class AdaptiveKVRecoveryNotImplementedError(AdaptiveKVError):
    """Raised when replay-based recovery is requested before it is implemented."""

