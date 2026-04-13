"""Layer 4 lifecycle and management state."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeLifecycle:
    """Layer 4-owned product lifecycle state."""

    model_id: str
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    ready_at: float | None = None
    stopped_at: float | None = None
    ready: bool = False
    stopped: bool = False
    last_error: str | None = None

    def mark_ready(self) -> None:
        self.ready_at = time.time()
        self.stopped_at = None
        self.ready = True
        self.stopped = False
        self.last_error = None

    def mark_stopped(self) -> None:
        self.stopped_at = time.time()
        self.stopped = True

    def mark_error(self, message: str) -> None:
        self.last_error = message

    def health_payload(
        self,
        *,
        metrics_enabled: bool,
        metrics_route_enabled: bool,
    ) -> dict[str, Any]:
        now = time.time()
        status = "ok" if self.ready and not self.stopped else "starting"
        if self.stopped:
            status = "stopped"
        payload: dict[str, Any] = {
            "status": status,
            "ready": self.ready and not self.stopped,
            "model_id": self.model_id,
            "instance_id": self.instance_id,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "stopped_at": self.stopped_at,
            "startup_duration_s": (
                max(0.0, self.ready_at - self.started_at) if self.ready_at is not None else None
            ),
            "uptime_s": max(0.0, now - self.started_at),
            "metrics_enabled": metrics_enabled,
            "metrics_route_enabled": metrics_route_enabled,
        }
        if self.last_error is not None:
            payload["last_error"] = self.last_error
        return payload
