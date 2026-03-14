"""Model protocol — contract for per-architecture model classes (§7, §9).

Every model in models/ must satisfy this protocol so that generate, batch,
and other consumers remain model-agnostic (AC17).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx

    from mlxs.protocols.cache import CacheProtocol


@runtime_checkable
class ModelProtocol(Protocol):
    """Structural contract for inference models (§7.3, §9).

    Models expose ``__call__`` for forward pass and ``make_cache`` to create
    compatible KV caches. The model must be an ``nn.Module`` subclass but
    this protocol captures only the inference-facing surface.
    """

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[CacheProtocol] | None = None,
        mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        """Run forward pass, return logits ``(B, T, V)``.

        Args:
            input_ids: Token ids ``(B, T)`` — batch x sequence length.
            cache: Per-layer KV cache objects. Updated in-place.
            mask: Attention mask ``(B, 1, T, S)`` or ``None`` for causal.
            input_embeddings: Pre-computed embeddings ``(B, T, D)``. When
                provided, used instead of ``embed_tokens(input_ids)`` (§7.4).

        Returns:
            Logits tensor of shape ``(B, T, vocab_size)``.
        """
        ...

    def make_cache(self) -> list[CacheProtocol]:
        """Create a list of empty KV caches, one per layer.

        The cache type (full, quantized, rotating) is determined by the model
        or by config — caller should not assume a specific type.
        """
        ...

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        ...

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        ...

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Remap or filter weight keys for this architecture.

        Called during model loading to adapt safetensors keys to the model's
        expected parameter names. Default: identity (return as-is).
        """
        ...

    def parameters(self) -> dict[str, Any]:
        """Return model parameters (nn.Module interface)."""
        ...
