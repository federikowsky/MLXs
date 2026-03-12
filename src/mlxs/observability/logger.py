"""Structured logging setup (NFR2, AC16).

Configures stdlib logging with a JSON formatter for structured output.
All modules use ``logging.getLogger(__name__)`` — this module only
configures the root logger and format.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        # Merge extra fields passed via logging.info("...", extra={...})
        for key in ("request_id", "model_id", "metric"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure the root logger for MLXs.

    Args:
        level: Logging level string (e.g. "INFO", "DEBUG").
        json_format: If True, use structured JSON output. If False, use
            stdlib default format (useful for local dev).
    """
    root = logging.getLogger("mlxs")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplication on re-init
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
        )
    root.addHandler(handler)
