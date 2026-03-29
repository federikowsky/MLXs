"""Discover locally cached Hugging Face text generation checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LocalModelInfo:
    """A single snapshot directory that looks like a loadable causal LM."""

    path: Path
    """Absolute path to the snapshot (contains config.json + weights)."""

    hub_repo_id: str
    """Inferred repo id from cache folder name (org--name)."""

    model_type: str
    """config.json model_type."""

    architectures: tuple[str, ...]
    """config.json architectures when present."""

    safetensors_files: tuple[str, ...]
    """Basenames of *.safetensors shards."""

    weight_bytes: int
    """Total size of ``*.safetensors`` under the snapshot (for sorting)."""


def _dir_to_repo_id(cache_name: str) -> str:
    if not cache_name.startswith("models--"):
        return cache_name
    inner = cache_name[len("models--") :]
    return inner.replace("--", "/", 1)


def infer_hub_repo_id(snapshot_dir: Path) -> str:
    """Resolve ``org/name`` from ``.../models--org--name/snapshots/<hash>``."""
    p = snapshot_dir.resolve()
    if p.parent.name == "snapshots" and p.parent.parent.name.startswith("models--"):
        return _dir_to_repo_id(p.parent.parent.name)
    if p.parent.name.startswith("models--"):
        return _dir_to_repo_id(p.parent.name)
    return p.name


def _is_causal_lm_arch(architectures: list[str]) -> bool:
    return any("CausalLM" in a or a.endswith("LMHeadModel") for a in architectures)


def _should_skip_config(config: dict[str, object], path: Path) -> str | None:
    arch = config.get("architectures")
    architectures: list[str] = arch if isinstance(arch, list) else []
    if architectures and not _is_causal_lm_arch(architectures):
        return "non_causal_architectures"
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        return "missing_model_type"
    # Hard skips: common non-LLM families in mixed caches
    skip_types = frozenset(
        {
            "bert",
            "xlm-roberta",
            "roberta",
            "clip",
            "wav2vec2",
            "whisper",
            "vit",
            "siglip",
            "autoencoder",
        }
    )
    if model_type.lower() in skip_types:
        return f"skipped_model_type:{model_type}"
    # Vision-heavy configs
    if any(k in config for k in ("vision_config", "audio_config")) and not _is_causal_lm_arch(
        architectures
    ):
        return "multimodal_without_causal_arch"
    if not list(path.glob("*.safetensors")):
        return "no_safetensors"
    return None


def discover_local_models(
    hub_root: Path | None = None,
    *,
    max_models: int | None = None,
) -> list[LocalModelInfo]:
    """Scan ``hub_root`` (default: ~/.cache/huggingface/hub) for causal LMs.

    Only directories matching ``models--*`` with a ``snapshots/*`` layout are
    considered. The newest snapshot by mtime is preferred when multiple exist.
    """
    root = hub_root or Path.home() / ".cache" / "huggingface" / "hub"
    if not root.is_dir():
        return []

    out: list[LocalModelInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("models--"):
            continue
        snapshots = child / "snapshots"
        if not snapshots.is_dir():
            continue
        snap_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
        if not snap_dirs:
            continue
        snap_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        chosen = snap_dirs[0]
        cfg_path = chosen / "config.json"
        if not cfg_path.is_file():
            continue
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        reason = _should_skip_config(config, chosen)
        if reason is not None:
            continue
        model_type = str(config["model_type"])
        arch_raw = config.get("architectures")
        arch_tuple = tuple(str(x) for x in arch_raw) if isinstance(arch_raw, list) else ()
        st_paths = list(chosen.glob("*.safetensors"))
        st = tuple(sorted(p.name for p in st_paths))
        wbytes = sum(p.stat().st_size for p in st_paths)
        repo_id = _dir_to_repo_id(child.name)
        out.append(
            LocalModelInfo(
                path=chosen.resolve(),
                hub_repo_id=repo_id,
                model_type=model_type,
                architectures=arch_tuple,
                safetensors_files=st,
                weight_bytes=wbytes,
            )
        )

    out.sort(key=lambda m: (m.weight_bytes, m.model_type, m.hub_repo_id))
    if max_models is not None:
        out = out[: max(0, max_models)]
    return out


def parse_explicit_model_paths(spec: str) -> list[Path]:
    """Parse comma-separated absolute or user-relative paths."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [Path(p).expanduser().resolve() for p in parts]
