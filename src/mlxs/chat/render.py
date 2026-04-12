"""Compatibility re-export for transcript presentation helpers.

The canonical presentation implementation now lives in ``mlxs.chat.present``.
Keep this module as a thin shim during the refactor so existing imports do not
break while callers migrate.
"""

from __future__ import annotations

from mlxs.chat.present.transcript import TranscriptEntry as _TranscriptEntry
from mlxs.chat.present.transcript import empty_state_fragments, entry_from_message, render_entry

__all__ = ["_TranscriptEntry", "entry_from_message", "render_entry", "empty_state_fragments"]
