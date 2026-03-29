"""Environment fingerprint for reproducible benchmark reports."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SystemFingerprint:
    python: str
    platform: str
    machine: str
    processor: str
    mlx_version: str | None
    mlx_lm_version: str | None
    mlxs_version: str | None
    transformers_version: str | None
    huggingface_hub_version: str | None
    sysctl_machdep_cpu_brand: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sysctl_cpu_brand() -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_fingerprint() -> SystemFingerprint:
    return SystemFingerprint(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        mlx_version=_pkg_version("mlx"),
        mlx_lm_version=_pkg_version("mlx-lm"),
        mlxs_version=_pkg_version("mlxs"),
        transformers_version=_pkg_version("transformers"),
        huggingface_hub_version=_pkg_version("huggingface-hub"),
        sysctl_machdep_cpu_brand=_sysctl_cpu_brand(),
    )


def fingerprint_json_line() -> str:
    return json.dumps(collect_fingerprint().to_json_dict(), sort_keys=True)
