"""Legacy generate protocol for compatibility surfaces above Layer 1.

The generate module exposes a function matching this protocol. It produces
a lazy stream of TokenEvent objects. No HTTP or server concepts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    import mlx.core as mx

    from mlxs._types import GenerateOptions, TokenEvent
    from mlxs.protocols.cache import CacheProtocol
    from mlxs.protocols.model import ModelProtocol


@runtime_checkable
class GenerateProtocol(Protocol):
    """Contract for the generate function (§9 "Interfaces").

    generate: model, tokenizer, prompt, options → stream of tokens.
    """

    def __call__(
        self,
        model: ModelProtocol,
        tokenizer: TokenizerProtocol,
        prompt: str | list[int],
        options: GenerateOptions,
        *,
        cache: list[CacheProtocol] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> Iterator[TokenEvent]:
        """Generate tokens from prompt.

        Args:
            model: Model to run inference on.
            tokenizer: Tokenizer for encoding/decoding.
            prompt: Input text or pre-tokenized ids.
            options: Generation parameters (temperature, top_p, etc.).
            cache: Optional pre-populated KV cache (e.g. from prompt cache).
            input_embeddings: Pre-computed embeddings ``(T, D)`` from
                multimodal preprocessing. When provided, used instead of
                ``embed_tokens`` during prefill (§7.4).

        Yields:
            TokenEvent for each generated token, with finish_reason set on
            the final event.
        """
        ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Minimal tokenizer contract needed by generate and batch.

    Wraps HuggingFace tokenizer; only the methods used by inference.
    """

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        ...

    def decode(self, token_ids: list[int] | int) -> str:
        """Decode token ids to text.

        Accepts a single int or a list of ints.
        """
        ...

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token id, if defined."""
        ...

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        ...
