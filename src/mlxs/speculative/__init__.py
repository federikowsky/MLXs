"""Speculative decoding — draft model + verify (§6.5, FR6, AC14).

Uses a smaller/faster draft model to generate candidate tokens,
then verifies them with the target model in a single forward pass.
Accepted tokens skip individual decode steps, improving throughput.
"""

from __future__ import annotations

from mlxs.speculative.draft import draft_tokens
from mlxs.speculative.verify import speculative_generate

__all__ = ["draft_tokens", "speculative_generate"]
