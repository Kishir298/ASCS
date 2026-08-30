"""Tests for the Ollama HTTP client using an in-process mock server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaModelNotFoundError,
    OllamaResponseError,
)


class _Handler(BaseHTTPRequestHandler):
    """Serves scripted responses. Behavior is driven by the server attrs."""

    status_map: dict = {}
    raw_body: bytes | None = None
    # Close each connection cleanly after one response. This avoids a
    # Windows keep-alive RST race between MockServer and urllib that would
    # otherwise intermittently abort reads with ConnectionAbortedError.
    protocol_version = "HTTP/1.0"

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > 0:
            try:
                self.rfile.read(length)
            except (OSError, ValueError):
                pass

    def _respond(self):
        status = type(self).status_map.get(self.path.split("?")[0], 200)
        if type(self).raw_body is not None:
            body = type(self).raw_body
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status = type(self).status_map.get(self.path.split("?")[0], 200)
        if status != 200:
            err = json.dumps({"error": f"http {status}"}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        body = b"{}"
        if self.path.startswith("/api/version"):
            body = json.dumps({"version": "0.3.0"}).encode()
        elif self.path.startswith("/api/tags"):
            body = json.dumps(
                {
                    "models": [
                        {"name": "qwen2.5-coder:7b"},
                        {"name": "deepseek-coder:6.7b"},
                    ]
                }
            ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._respond()

    def do_POST(self):
        self._read_body()
        self._respond()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    """A running mock server; yields a (base_url, start, stop) helper."""
    import types

    handler = type("Handler", (_Handler,), {"status_map": {}, "raw_body": None})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"

    def set(**kwargs):
        handler.status_map = kwargs.get("status", {})
        handler.raw_body = kwargs.get("body")

    yield types.SimpleNamespace(base=base, set=set, handler=handler)
    httpd.shutdown()
    httpd.server_close()


def test_connectivity_ok(server):
    client = OllamaClient(base_url=server.base)
    assert client.check_connectivity(timeout=5) is True
    client.check_connectivity(timeout=5)


def test_list_models(server):
    client = OllamaClient(base_url=server.base)
    assert client.list_models() == [
        "qwen2.5-coder:7b",
        "deepseek-coder:6.7b",
    ]


def test_is_model_available(server):
    client = OllamaClient(base_url=server.base)
    assert client.is_model_available("qwen2.5-coder:7b") is True
    assert client.is_model_available("unknown") is False


def test_chat_single_json(server):
    server.set(
        body=json.dumps(
            {"message": {"role": "assistant", "content": '{"tool":"read_file","arguments":{"path":"a.py"}}'}, "done": True}
        ).encode()
    )
    client = OllamaClient(base_url=server.base)
    out = client.chat([{"role": "user", "content": "hi"}])
    assert "read_file" in out


def test_chat_ndjson(server):
    # Some Ollama builds stream NDJSON lines even with stream:false. Each line
    # is a complete JSON object contributing a content delta.
    lines = [
        {"message": {"content": '{"co'}},
        {"message": {"content": 'mment":"hi","tool'}},
        {"message": {"content": '":"list_directory","arguments":{"path":"."}}'}},
        {"done": True},
    ]
    server.set(body=("\n".join(json.dumps(l) for l in lines) + "\n").encode())
    client = OllamaClient(base_url=server.base)
    out = client.chat([{"role": "user", "content": "hi"}])
    assert '{"comment":"hi","tool":"list_directory","arguments":{"path":"."}}' in out


def test_chat_model_not_found(server):
    server.set(status={"/api/chat": 404})
    client = OllamaClient(base_url=server.base)
    with pytest.raises(OllamaModelNotFoundError):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_generic_http_error(server):
    server.set(status={"/api/chat": 500})
    client = OllamaClient(base_url=server.base)
    # A generic non-404 HTTP error must NOT be misreported as model-not-found.
    with pytest.raises(OllamaHTTPError):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_connection_refused():
    client = OllamaClient(base_url="http://127.0.0.1:1", request_timeout=2)
    with pytest.raises(OllamaConnectionError):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_empty_body(server):
    server.set(body=b"")
    client = OllamaClient(base_url=server.base)
    with pytest.raises(OllamaResponseError):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_garbage_body(server):
    server.set(body=b"this is not json at all {")
    client = OllamaClient(base_url=server.base)
    with pytest.raises(OllamaResponseError):
        client.chat([{"role": "user", "content": "hi"}])
