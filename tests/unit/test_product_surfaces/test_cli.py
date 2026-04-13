from __future__ import annotations

import socket

import pytest

from mlxs._errors import MLXsError
from mlxs.product_surfaces.cli import _assert_server_endpoint_available


def test_assert_server_endpoint_available_accepts_free_port() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()

    _assert_server_endpoint_available(host, port)


def test_assert_server_endpoint_available_rejects_in_use_port() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    try:
        with pytest.raises(MLXsError, match="already in use"):
            _assert_server_endpoint_available(host, port)
    finally:
        sock.close()
