"""Error hierarchy for MLXs (§6.9).

All exceptions are HTTP-agnostic. The server module maps them to status codes.
Each exception carries a `status_hint` used by the server for HTTP responses.
"""

from __future__ import annotations


class MLXsError(Exception):
    """Base exception for all MLXs errors."""

    status_hint: int = 500

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


# --- Model / load errors (500) ---


class ModelLoadError(MLXsError):
    """Model weights or config could not be loaded (§6.9)."""

    status_hint = 500


class TokenizerError(MLXsError):
    """Tokenizer could not be loaded or applied."""

    status_hint = 500


class ModelNotLoadedError(MLXsError):
    """Operation requires a loaded model but none is available."""

    status_hint = 503


# --- Request errors (4xx) ---


class InvalidPromptError(MLXsError):
    """Prompt is invalid: empty, exceeds context window, etc. (§6.9)."""

    status_hint = 400


class InvalidConfigError(MLXsError):
    """Configuration option is invalid or unknown (§8.1)."""

    status_hint = 400


# --- Capacity / resource errors (503) ---


class MemoryCeilingError(MLXsError):
    """Memory ceiling exceeded and cannot be resolved by trimming (§6.3.1, §6.9)."""

    status_hint = 503


class CapacityExceededError(MLXsError):
    """Server at capacity: queue full or max concurrent requests reached (§6.7)."""

    status_hint = 503


class RequestTimeoutError(MLXsError):
    """Request exceeded configured timeout (§6.9)."""

    status_hint = 503


# --- Generation errors ---


class GenerationError(MLXsError):
    """Error during token generation."""

    status_hint = 500


class StopSequenceError(MLXsError):
    """Internal signal: stop sequence matched. Not raised to callers."""

    status_hint = 200
