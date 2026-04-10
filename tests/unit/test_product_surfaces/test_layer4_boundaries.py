"""Static boundary checks for Phase 5 Layer 4 attachment."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_cli_entrypoint_delegates_to_product_surfaces() -> None:
    source = _read("src/mlxs/cli.py")
    assert "from mlxs.product_surfaces.cli import main" in source


def test_server_app_delegates_to_product_surfaces_http() -> None:
    source = _read("src/mlxs/server/app.py")
    assert "from mlxs.product_surfaces.http import create_app" in source


def test_server_deps_delegates_to_bootstrap() -> None:
    source = _read("src/mlxs/server/deps.py")
    assert "from mlxs.product_surfaces.bootstrap import ProductRuntime as Dependencies" in source
    assert "return create_runtime(config)" in source
    assert "load_model_and_tokenizer" not in source
    assert "PromptCache(" not in source
    assert "create_metrics" not in source


def test_bootstrap_is_only_concrete_wiring_site() -> None:
    source = _read("src/mlxs/product_surfaces/bootstrap.py")
    assert "load_model_and_tokenizer" in source
    assert "PromptCache(" in source
    assert "PromptCacheOrchestrator(" in source
    assert "RequestQueue(" in source
    assert "generate_single_request" in source
    assert "generate_compat" not in source


def test_http_surface_no_longer_uses_deps_alias() -> None:
    source = _read("src/mlxs/product_surfaces/http.py")
    assert "app.state.deps" not in source
    assert "request.app.state.runtime" in source


def test_server_chat_wrapper_not_canonical_implementation() -> None:
    source = _read("src/mlxs/server/chat.py")
    assert "from mlxs.product_surfaces.chat import (" in source
    assert "Layer 4 compatibility wrapper" in source


def test_canonical_chat_surface_has_no_prompt_cache_fallback_bridge() -> None:
    source = _read("src/mlxs/product_surfaces/chat.py")
    assert "prompt_cache_orchestrator = getattr" not in source
    assert "PromptCacheOrchestrator(" not in source
    assert "deps.prompt_cache_orchestrator.prepare" in source


def test_benchmark_path_stays_free_of_product_surfaces() -> None:
    source = _read("benchmarks/mlxs_vs_mlx_lm/backends.py")
    assert "product_surfaces" not in source


def test_lower_layers_do_not_import_product_surfaces() -> None:
    assert "product_surfaces" not in _read("src/mlxs/general_path/single_request.py")
    assert "product_surfaces" not in _read("src/mlxs/advanced_engines/prompt_cache.py")
