"""Opt-in live tests against a real Ollama server.

These tests exercise the real HTTP client against a running Ollama instance and
only run when ``RISALIVE=1`` is set. They never run during an ordinary
``pytest -q`` run, so normal CI / local development is never blocked on a live
model or server.

Run them with:

    RISALIVE=1 pytest -q tests/test_ollama_live.py

Requirements to actually execute (not just collect):
    * Ollama is running at OLLAMA_BASE_URL (default http://localhost:11434).
    * The configured model (default qwen3-coder:30b, fallback qwen2.5-coder:14b) is installed, e.g.
      ``ollama pull qwen3-coder:30b``.
"""

from __future__ import annotations

import os

import pytest

from agent.config import DEFAULT_MODEL
from agent.ollama import (
    OllamaClient,
    OllamaModelNotFoundError,
)

RISALIVE = os.environ.get("RISALIVE") == "1"

pytestmark = pytest.mark.skipif(
    not RISALIVE,
    reason="Live Ollama test - opt-in: set RISALIVE=1",
)

LIVE_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LIVE_MODEL = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def _client() -> OllamaClient:
    return OllamaClient(
        base_url=LIVE_BASE_URL,
        model=LIVE_MODEL,
        request_timeout=180,
        keep_alive="1m",
    )


def test_live_ollama_reachable_and_model_installed():
    """Connectivity, version detection, and model availability."""
    client = _client()
    report = client.ensure_ready(check_timeout=5, prewarm=False)
    assert report["reachable"] is True, "Ollama not reachable"
    assert report["version"], "Ollama version missing"
    assert report["available"] is True, (
        f"model {LIVE_MODEL!r} not installed; run: ollama pull {LIVE_MODEL}"
    )


def test_live_ollama_model_missing():
    """A model that is not installed must surface as model-not-found."""
    client = OllamaClient(base_url=LIVE_BASE_URL, model="definitely-not-a-model:0")
    with pytest.raises(OllamaModelNotFoundError):
        client.chat([{"role": "user", "content": "hi"}])


def test_live_ollama_tiny_chat_non_empty():
    """A tiny prompt must produce a non-empty assistant response.

    Note: the per-call ceiling is generous (180 s) because non-streaming
    generation is CPU-bound on slower machines — a warm ``qwen3-coder:30b`` on a
    laptop routinely takes 40+ s to finish a short answer here.
    """
    client = _client()
    out = client.chat(
        [{"role": "user", "content": "Reply with exactly: OPENCODE LOCAL QWEN TEST OK"}],
        format=None,
        timeout=180,
    )
    assert out.strip(), "chat returned an empty response"


def test_live_ollama_stream_non_empty():
    """Streaming chat must yield at least one visible token."""
    client = _client()
    chunks = list(
        client.chat_stream(
            [{"role": "user", "content": "Say the single word: hello"}],
            format=None,
            timeout=30,
        )
    )
    joined = "".join(chunks).strip()
    assert joined, "stream produced no visible output"
