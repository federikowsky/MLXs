"""Model path resolution — local directory or Hugging Face id (FR1, §7.2).

Resolves model_path to a local directory Path. If the path is not an existing
directory, treats it as a Hugging Face repo id and downloads via snapshot_download.
All loaders then receive a local Path. Never logs or exposes the token.
"""

from __future__ import annotations

from pathlib import Path

from mlxs._errors import ModelLoadError


def resolve_model_path(
    model_path: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Resolve model path to a local directory Path.

    If model_path is an existing directory, returns it as Path. Otherwise
    downloads the repo from Hugging Face via snapshot_download (using
    optional revision and token) and returns the cache path. Never logs token.

    Raises:
        ModelLoadError: If path is neither a local directory nor a valid HF id.
    """
    path = Path(model_path)
    if path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelLoadError(
            "Resolving Hugging Face model id requires 'huggingface_hub'. "
            "Install with: pip install huggingface_hub"
        ) from exc
    kwargs: dict[str, str | None] = {}
    if revision is not None:
        kwargs["revision"] = revision
    if token is not None:
        kwargs["token"] = token
    try:
        resolved = snapshot_download(str(model_path), **kwargs)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to resolve model path (use a local directory or a valid Hugging Face model id): {exc}"
        ) from exc
    return Path(resolved)
