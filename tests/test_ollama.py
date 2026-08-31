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
    OllamaTimeoutError,
    _strip_think_tags,
    _strip_think_tags_streaming,
)


class _Handler(BaseHTTPRequestHandler):
    """Serves scripted responses. Behavior is driven by the server attrs."""

    status_map: dict = {}
    raw_body: bytes | None = None
    captured_body: bytes | None = None
    # Close each connection cleanly after one response. This avoids a
    # Windows keep-alive RST race between MockServer and urllib that would
    # otherwise intermittently abort reads with ConnectionAbortedError.
    protocol_version = "HTTP/1.0"

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > 0:
            try:
                type(self).captured_body = self.rfile.read(length)
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
        handler.captured_body = None

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


def test_chat_sends_keep_alive_from_instance(server):
    server.set(
        body=json.dumps({"message": {"content": "ok"}, "done": True}).encode()
    )
    client = OllamaClient(base_url=server.base, keep_alive="30m")
    client.chat([{"role": "user", "content": "hi"}])
    payload = json.loads(server.handler.captured_body.decode())
    assert payload["keep_alive"] == "30m"
    assert payload["stream"] is False
    assert payload["model"] == client.model


def test_chat_keep_alive_call_arg_overrides_instance(server):
    server.set(
        body=json.dumps({"message": {"content": "ok"}, "done": True}).encode()
    )
    client = OllamaClient(base_url=server.base, keep_alive="1m")
    client.chat([{"role": "user", "content": "hi"}], keep_alive="10m")
    payload = json.loads(server.handler.captured_body.decode())
    assert payload["keep_alive"] == "10m"


def test_chat_no_keep_alive_key_when_unset(server):
    server.set(
        body=json.dumps({"message": {"content": "ok"}, "done": True}).encode()
    )
    client = OllamaClient(base_url=server.base, keep_alive=None)
    client.chat([{"role": "user", "content": "hi"}])
    payload = json.loads(server.handler.captured_body.decode())
    assert "keep_alive" not in payload


def test_ensure_ready_report(server):
    client = OllamaClient(base_url=server.base, model="qwen2.5-coder:7b")
    report = client.ensure_ready(check_timeout=5, prewarm=False)
    assert report["reachable"] is True
    assert report["available"] is True
    assert "qwen2.5-coder:7b" in report["installed"]
    assert report["version"] == "0.3.0"
    assert report["warmed"] is False


def test_ensure_ready_prewarm_false_avoids_warm(server):
    client = OllamaClient(base_url=server.base, model="qwen2.5-coder:7b")
    report = client.ensure_ready(check_timeout=5, prewarm=False)
    assert report["warmed"] is False


class _FakeResp:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_abort_current_closes_active_response():
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    resp = _FakeResp()
    assert client._active is None
    client._track(resp, 60)
    assert client._active is resp
    client.abort_current()
    assert resp.closed == 1
    assert client._active is None


def test_abort_current_without_active_is_noop():
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    client.abort_current()  # must not raise
    assert client._active is None


def test_untrack_ignores_other_response():
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    a = _FakeResp()
    b = _FakeResp()
    client._track(a, 60)
    client._untrack(b)  # different resp: must keep tracking a
    assert client._active is a
    client._untrack(a)
    assert client._active is None


# ── chat_resilient: bounded retry on transient failures ─────────────────────

class _FlakyChat(OllamaClient):
    """A client whose ``chat`` raises transient errors N times then succeeds."""

    def __init__(self, fail_kind, fails_before_success, model="fake"):
        super().__init__(base_url="http://127.0.0.1:1", model=model)
        self.fail_kind = fail_kind
        self.fails_before_success = fails_before_success
        self.calls = 0

    def chat(self, messages, *, format="json", options=None, timeout=None):
        self.calls += 1
        if self.calls <= self.fails_before_success:
            if self.fail_kind == "conn":
                raise OllamaConnectionError("cannot reach")
            if self.fail_kind == "timeout":
                raise OllamaTimeoutError("timed out")
            if self.fail_kind == "http500":
                raise OllamaHTTPError(500, "server error")
            if self.fail_kind == "http404":
                raise OllamaHTTPError(404, "not found")
        return '{"done": true, "summary": "ok"}'


def test_chat_resilient_recovers_after_connection_errors():
    client = _FlakyChat("conn", fails_before_success=2)
    out = client.chat_resilient([{"role": "user", "content": "hi"}], max_retries=2, backoff_s=0)
    assert out == '{"done": true, "summary": "ok"}'
    assert client.calls == 3


def test_chat_resilient_recovers_after_http500():
    client = _FlakyChat("http500", fails_before_success=1)
    out = client.chat_resilient([{"role": "user", "content": "hi"}], max_retries=2, backoff_s=0)
    assert client.calls == 2
    assert "done" in out


def test_chat_resilient_raises_when_retries_exhausted():
    client = _FlakyChat("conn", fails_before_success=99)
    with pytest.raises(OllamaConnectionError):
        client.chat_resilient([{"role": "user", "content": "hi"}], max_retries=2, backoff_s=0)
    # 1 initial + 2 retries
    assert client.calls == 3


def test_chat_resilient_does_not_retry_non_transient_http404():
    client = _FlakyChat("http404", fails_before_success=99)
    with pytest.raises(OllamaHTTPError) as exc:
        client.chat_resilient([{"role": "user", "content": "hi"}], max_retries=2, backoff_s=0)
    assert exc.value.status == 404
    assert client.calls == 1  # no retry for 4xx


def test_chat_resilient_respects_should_stop():
    client = _FlakyChat("conn", fails_before_success=99)
    with pytest.raises(OllamaConnectionError):
        client.chat_resilient(
            [{"role": "user", "content": "hi"}],
            max_retries=5, backoff_s=0,
            should_stop=lambda: True,
        )
    assert client.calls == 1  # stopped before retrying


def test_chat_resilient_fallback_for_plain_chat_clients():
    """Clients without chat_resilient still work through loop/executor helper."""
    client = _FlakyChat("conn", fails_before_success=0)
    from agent.loop import _resilient_chat

    out = _resilient_chat(client, [{"role": "user", "content": "hi"}], format="json")
    assert out == '{"done": true, "summary": "ok"}'


# ── think-tag stripping ────────────────────────────────────────────────────

def test_strip_think_tags_removes_block():
    text = '<think>Let me think...</think>\n{"tool": "read_file"}'
    result = _strip_think_tags(text)
    assert "<think>" not in result
    assert "</think>" not in result
    assert '{"tool": "read_file"}' in result


def test_strip_think_tags_multiple_blocks():
    text = '<think>first</think><think>second</think>done'
    result = _strip_think_tags(text)
    assert "<think>" not in result
    assert result == "done"


def test_strip_think_tags_no_tags_unchanged():
    text = '{"done": true, "summary": "ok"}'
    assert _strip_think_tags(text) == text


def test_strip_think_tags_empty():
    assert _strip_think_tags("") == ""


def test_strip_think_tags_multiline():
    text = "<think>\nline 1\nline 2\n</think>\nresult"
    result = _strip_think_tags(text)
    assert "<think>" not in result
    assert "line 1" not in result
    assert result == "result"


def test_strip_think_tags_streaming_no_tags():
    vis, buf = _strip_think_tags_streaming("", "hello world")
    assert vis == "hello world"
    assert buf == ""


def test_strip_think_tags_streaming_complete_in_one_delta():
    vis, buf = _strip_think_tags_streaming(
        "", "<think>thinking</think>done"
    )
    assert vis == "done"
    assert buf == ""


def test_strip_think_tags_streaming_opening_only_buffers():
    vis, buf = _strip_think_tags_streaming("", "<think>let me think")
    assert vis == ""
    assert "<think>" in buf


def test_strip_think_tags_streaming_closing_after_buffer():
    vis1, buf1 = _strip_think_tags_streaming("", "<think>let me think")
    assert vis1 == ""
    vis2, buf2 = _strip_think_tags_streaming(buf1, " about this...</think>result")
    assert vis2 == "result"
    assert buf2 == ""


def test_strip_think_tags_streaming_normal_content_before_think():
    vis, buf = _strip_think_tags_streaming("", "hello <think>thinking")
    assert vis == "hello"
    assert "<think>" in buf


def test_strip_think_tags_streaming_two_round_trips():
    """Full think block across 3 deltas: pre-text, think, post-text."""
    vis, buf = _strip_think_tags_streaming("", "step 1")
    assert vis == "step 1"
    assert buf == ""
    vis, buf = _strip_think_tags_streaming(buf, "<think>reasoning")
    assert vis == ""
    assert "<think>" in buf
    vis, buf = _strip_think_tags_streaming(buf, " more</think>final")
    assert vis == "final"
    assert buf == ""

