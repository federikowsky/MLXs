"""Vision encoders and utilities for multimodal models (FR12, §7.4).

Provides merge_embeddings() for fusing vision/audio features into text
embedding sequences at placeholder token positions.
"""

from __future__ import annotations

import mlx.core as mx

from mlxs._errors import InvalidPromptError


def merge_embeddings(
    text_embeddings: mx.array,
    media_embeddings: mx.array,
    input_ids: mx.array,
    placeholder_token_id: int,
) -> mx.array:
    """Replace placeholder token embeddings with media embeddings.

    Args:
        text_embeddings: ``(B, T, D)`` from ``embed_tokens(input_ids)``.
        media_embeddings: ``(N, D)`` output of encoder + projector.
        input_ids: ``(B, T)`` to find placeholder positions.
        placeholder_token_id: Token id marking media positions (e.g. 151655).

    Returns:
        ``(B, T, D)`` with media embeddings at placeholder positions.

    Raises:
        InvalidPromptError: If placeholder count != media embedding count.
    """
    B = input_ids.shape[0]
    batch_outputs = []
    feature_idx = 0

    for b in range(B):
        mask = input_ids[b] == placeholder_token_id
        num_positions = int(mx.sum(mask).item())

        if num_positions > 0:
            batch_features = media_embeddings[feature_idx : feature_idx + num_positions]
            if batch_features.shape[0] != num_positions:
                raise InvalidPromptError(
                    f"Expected {num_positions} media tokens but got "
                    f"{batch_features.shape[0]} media embeddings"
                )
            # Build indices: for placeholder positions, index into batch_features
            cumsum = mx.cumsum(mask.astype(mx.int32))
            feature_indices = mx.where(mask, cumsum - 1, 0)
            gathered = batch_features[feature_indices]
            mask_expanded = mx.expand_dims(mask, axis=-1)
            batch_output = mx.where(mask_expanded, gathered, text_embeddings[b])
            feature_idx += num_positions
        else:
            batch_output = text_embeddings[b]

        batch_outputs.append(batch_output)

    return mx.stack(batch_outputs, axis=0)


__all__ = ["merge_embeddings"]
