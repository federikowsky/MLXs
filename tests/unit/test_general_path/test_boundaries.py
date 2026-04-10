"""Static boundary checks for Phase 3 Layer 2 attachment."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_generate_alias_routes_to_general_path_without_generate_compat() -> None:
    source = _read("src/mlxs/generate/__init__.py")
    assert "from mlxs.general_path import generate_single_request" in source
    assert "return generate_single_request(" in source
    assert "def generate_compat" not in source
    assert "decode_loop" not in source
    assert "chunked_prefill" not in source


def test_general_path_consumes_runtime_core_not_upper_layers() -> None:
    source = _read("src/mlxs/general_path/single_request.py")
    assert "from mlxs.runtime_core" in source
    assert "from mlxs.server" not in source
    assert "from mlxs.batch" not in source
    assert "from mlxs.speculative" not in source


def test_canonical_benchmark_path_stays_on_runtime_core() -> None:
    source = _read("benchmarks/mlxs_vs_mlx_lm/backends.py")
    assert "from mlxs.runtime_core" in source
    assert "from mlxs.generate" not in source
