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
    python_version: str
    platform: str
    machine: str
    processor: str
    macos_version: str | None
    soc: str | None
    ram_bytes: int | None
    mlx_version: str | None
    mlx_lm_version: str | None
    mlxs_version: str | None
    transformers_version: str | None
    huggingface_hub_version: str | None
    mlxs_git_revision: str | None
    mlxs_git_branch: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sysctl_value(key: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key],
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


def _git_value(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return value or None


def _sysctl_memsize_bytes() -> int | None:
    raw = _sysctl_value("hw.memsize")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def collect_fingerprint() -> SystemFingerprint:
    return SystemFingerprint(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        macos_version=platform.mac_ver()[0] or None,
        soc=_sysctl_value("machdep.cpu.brand_string") or platform.processor() or None,
        ram_bytes=_sysctl_memsize_bytes(),
        mlx_version=_pkg_version("mlx"),
        mlx_lm_version=_pkg_version("mlx-lm"),
        mlxs_version=_pkg_version("mlxs"),
        transformers_version=_pkg_version("transformers"),
        huggingface_hub_version=_pkg_version("huggingface-hub"),
        mlxs_git_revision=_git_value("rev-parse", "HEAD"),
        mlxs_git_branch=_git_value("branch", "--show-current"),
    )


def fingerprint_json_line() -> str:
    return json.dumps(collect_fingerprint().to_json_dict(), sort_keys=True)
