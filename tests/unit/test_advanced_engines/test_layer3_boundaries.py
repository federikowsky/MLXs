"""Static boundary checks for Phase 4 Layer 3 attachment."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_batch_package_no_longer_exports_canonical_alias() -> None:
    source = _read("src/mlxs/batch/__init__.py")
    assert "BatchScheduler" not in source
    assert "__all__: list[str] = []" in source


def test_speculative_package_no_longer_exports_canonical_alias() -> None:
    source = _read("src/mlxs/speculative/__init__.py")
    assert "speculative_generate" not in source
    assert "draft_tokens" not in source
    assert "__all__: list[str] = []" in source


def test_server_chat_is_only_a_layer4_wrapper() -> None:
    source = _read("src/mlxs/server/chat.py")
    assert "from mlxs.product_surfaces.chat import (" in source
    assert "run_chat_loop" in source
    assert "_handle_plain_command" in source
    assert "PromptCacheOrchestrator" not in source


def test_advanced_engines_stay_below_product_surfaces() -> None:
    source = _read("src/mlxs/advanced_engines/prompt_cache.py")
    assert "from mlxs.server" not in source
    assert "from starlette" not in source


def test_canonical_benchmark_path_stays_free_of_layer3() -> None:
    source = _read("benchmarks/mlxs_vs_mlx_lm/backends.py")
    assert "from mlxs.advanced_engines" not in source
