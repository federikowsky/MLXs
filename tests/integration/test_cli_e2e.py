"""End-to-end CLI tests — subprocess invocations of mlxs.

No model required for basic tests. Model-dependent tests use MLXS_E2E_MODEL.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError

import pytest


def _run_mlxs(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    """Run mlxs CLI; entry point from installed package or -m."""
    cmd = [sys.executable, "-m", "mlxs.cli"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )


class TestCLIHelp:
    """CLI responds to --help without model."""

    def test_mlxs_help(self) -> None:
        r = _run_mlxs(["--help"])
        assert r.returncode == 0
        assert "serve" in r.stdout and "chat" in r.stdout

    def test_serve_help(self) -> None:
        r = _run_mlxs(["serve", "--help"])
        assert r.returncode == 0
        assert "config" in r.stdout or "port" in r.stdout

    def test_chat_help(self) -> None:
        r = _run_mlxs(["chat", "--help"])
        assert r.returncode == 0
        assert "config" in r.stdout or "model" in r.stdout


class TestCLIExitCodes:
    """Subcommand and invalid args."""

    def test_no_subcommand_exits_nonzero(self) -> None:
        r = _run_mlxs([])
        assert r.returncode != 0

    def test_serve_with_invalid_model_exits_nonzero(self) -> None:
        """Serve with invalid model path fails (config resolve + load)."""
        r = _run_mlxs(["serve", "--model", "/nonexistent/model/path/12345"])
        assert r.returncode != 0


@pytest.mark.integration
class TestServeE2E:
    """Full server start + health check; requires MLXS_E2E_MODEL."""

    @pytest.fixture
    def e2e_model(self) -> str | None:
        return os.environ.get("MLXS_E2E_MODEL")

    def test_serve_health_with_model(self, e2e_model: str | None) -> None:
        if not e2e_model:
            pytest.skip("MLXS_E2E_MODEL not set")
        proc = subprocess.Popen(
            [sys.executable, "-m", "mlxs.cli", "serve", "--model", e2e_model, "--port", "19999"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        try:
            for _ in range(30):
                time.sleep(0.5)
                try:
                    with urlopen("http://127.0.0.1:19999/health", timeout=2) as resp:
                        if resp.status == 200:
                            return
                except (URLError, OSError):
                    pass
            pytest.fail("Server did not respond on /health within 15s")
        finally:
            proc.terminate()
            proc.wait(timeout=5)
