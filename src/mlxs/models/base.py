"""Base for per-architecture model args (§7, AC17).

Provides BaseModelArgs for config deserialization. Mask and SDPA helpers
live in cache (attention_mask) and layers (attention) respectively.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseModelArgs:
    """Base for per-architecture model args.

    Provides ``from_dict`` that filters unknown keys, allowing forward
    compatibility with newer config.json fields.
    """

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> BaseModelArgs:
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})
