"""Stream policy — single vs overlap MLX streams (§6.8, AC12).

Provides stream context managers for controlling MLX stream assignment
during prefill and decode. Overlap mode uses separate streams for
prefill and decode to allow pipelining.

Fallback-safe: single stream is the default (AC12).
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator

import mlx.core as mx

from mlxs._types import StreamPolicy

# Default MLX stream for generation
_generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def generation_stream_context(
    policy: StreamPolicy = StreamPolicy.SINGLE,
) -> Generator[mx.Stream, None, None]:
    """Context manager for the generation MLX stream.

    Args:
        policy: Stream policy — SINGLE uses default stream,
            OVERLAP uses a dedicated generation stream.

    Yields:
        The MLX stream to use for generation operations.
    """
    if policy == StreamPolicy.OVERLAP:
        with mx.stream(_generation_stream):
            yield _generation_stream
    else:
        # Single stream — use default device stream
        yield mx.default_stream(mx.default_device())
