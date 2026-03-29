"""Process RSS helpers (macOS vs Linux semantics)."""

from __future__ import annotations

import resource
import sys


def rss_bytes_self() -> int:
    """Current process max resident set size from getrusage.

    macOS: ``ru_maxrss`` is bytes. Linux: kilobytes (multiply by 1024).
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024
