"""Layer 4 lifecycle and management state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeLifecycle:
    """Layer 4-owned product lifecycle state."""

    model_id: str
    started_at: float = field(default_factory=time.time)
    ready: bool = False
    stopped: bool = False
    last_error: str | None = None

    def mark_ready(self) -> None:
        self.ready = True
        self.stopped = False
        self.last_error = None

    def mark_stopped(self) -> None:
        self.stopped = True

    def mark_error(self, message: str) -> None:
        self.last_error = message

    def health_payload(
        self,
        *,
        metrics_enabled: bool,
        metrics_route_enabled: bool,
    ) -> dict[str, Any]:
        status = "ok" if self.ready and not self.stopped else "starting"
        if self.stopped:
            status = "stopped"
        payload: dict[str, Any] = {
            "status": status,
            "ready": self.ready and not self.stopped,
            "model_id": self.model_id,
            "started_at": self.started_at,
            "metrics_enabled": metrics_enabled,
            "metrics_route_enabled": metrics_route_enabled,
        }
        if self.last_error is not None:
            payload["last_error"] = self.last_error
        return payload
