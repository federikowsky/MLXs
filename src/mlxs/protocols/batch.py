"""Batch scheduler protocol — contract for batched inference (§6.4, §9).

The batch module implements continuous batching. Server depends only on
this protocol for dispatching concurrent requests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mlxs._types import GenerateOptions, TokenEvent
    from mlxs.protocols.generate import TokenizerProtocol
    from mlxs.protocols.model import ModelProtocol


@runtime_checkable
class BatchSchedulerProtocol(Protocol):
    """Contract for the batch scheduler (§6.4, §9).

    Manages concurrent generation sequences. Server adds requests and
    iterates results; the scheduler handles padding, batching, and
    continuous insertion/removal of sequences.
    """

    def add(
        self,
        request_id: str,
        model: ModelProtocol,
        tokenizer: TokenizerProtocol,
        prompt: str | list[int],
        options: GenerateOptions,
    ) -> None:
        """Enqueue a new generation request.

        Args:
            request_id: Unique identifier for this request.
            model: Model to run inference on.
            tokenizer: Tokenizer for encoding/decoding.
            prompt: Input text or pre-tokenized ids.
            options: Generation parameters.
        """
        ...

    def remove(self, request_id: str) -> None:
        """Cancel and remove a request from the batch.

        If the request has already finished, this is a no-op.
        """
        ...

    def step(self) -> dict[str, list[TokenEvent]]:
        """Run one batch step (prefill or decode) for all active sequences.

        Returns:
            Mapping of request_id → list of new TokenEvent(s) produced
            in this step. Empty dict if no active sequences.
        """
        ...

    def drain(self) -> Iterator[tuple[str, list[TokenEvent]]]:
        """Drain all finished sequences from the batch.

        Yields:
            Tuples of (request_id, all_token_events) for completed sequences.
        """
        ...

    @property
    def active_count(self) -> int:
        """Number of currently active sequences in the batch."""
        ...

    @property
    def pending_count(self) -> int:
        """Number of requests waiting to enter the batch."""
        ...
