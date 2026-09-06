"""Readable model/provider error contract tests (no live Ollama)."""

from __future__ import annotations

import pytest

from agent.tui import format_task_error

PROVIDER = "ollama"
MODEL = "qwen3-coder:30b"


def _message(status="", summary="", error=""):
    return format_task_error(status, summary, error, PROVIDER, MODEL)


def _assert_clean(text):
    assert "Traceback (most recent call last)" not in text
    assert 'File "' not in text
    assert "urllib3" not in text
    assert "requests.adapters" not in text


@pytest.mark.parametrize(
    ("status", "summary", "error", "markers"),
    [
        (
            "fatal",
            "Ollama is unavailable.",
            "cannot reach http://localhost:11434: refused",
            ["ollama", "connection", "qwen3-coder:30b", "ollama serve"],
        ),
        (
            "fatal",
            "Ollama request timed out.",
            "request timed out after 600s",
            ["ollama", "timed out", "qwen3-coder:30b", "Retrying may succeed"],
        ),
        (
            "fatal",
            "Model 'qwen3-coder:30b' is not installed on the Ollama server.",
            "Ollama HTTP 404: model not found",
            ["not available", "qwen3-coder:30b", "ollama pull", "/models"],
        ),
        (
            "fatal",
            "Ollama request failed.",
            "Ollama HTTP 500: internal server error",
            ["server failed", "retried", "qwen3-coder:30b"],
        ),
        (
            "malformed",
            "Model repeatedly failed to produce a valid response.",
            "expected tool call",
            ["could not use", "qwen3-coder:30b", "Retry"],
        ),
        (
            "fatal",
            "Model returned an empty assistant response.",
            "",
            ["could not use", "qwen3-coder:30b"],
        ),
        (
            "fatal",
            "Worker crash: boom",
            "boom",
            ["unexpectedly", "usable"],
        ),
        (
            "fatal",
            "",
            "",
            ["did not complete", "qwen3-coder:30b", "/check"],
        ),
    ],
)
def test_error_matrix_is_readable(status, summary, error, markers):
    text = _message(status, summary, error)
    _assert_clean(text)
    for marker in markers:
        assert marker in text


def test_budget_timeout_names_the_budget():
    text = _message(
        "max_iterations",
        "Stopped after 50 iterations (AGENT_MAX_ITERATIONS=50).",
        "",
    )
    assert "iteration budget" in text
    assert "AGENT_MAX_ITERATIONS" in text
    assert "Completed" not in text


def test_cancelled_stays_cancelled():
    assert _message("cancelled", "Stopped by the operator.", "") == "Cancelled."


def test_raw_detail_is_truncated():
    text = _message("fatal", "x" * 2000, "")
    assert len(text) < 2000
    assert "…" in text
    _assert_clean(text)


def test_empty_everything_still_readable():
    text = format_task_error("", "", "", "", "")
    assert "did not complete" in text
    assert "Provider: ollama" in text
    _assert_clean(text)


def test_no_false_success_words_for_failures():
    for status in ("fatal", "malformed", "max_iterations", "failed"):
        text = _message(status, "something broke", "boom")
        assert "Completed — ready" not in text
        assert text.startswith("Model error")


# -- terminal result contract (fake hub/runner, no live Ollama) -----------------


class _FakeEvent:
    def __init__(self, type="", message=""):
        self.type = type
        self.message = message


class _FakeHub:
    def __init__(self, events=()):
        self._events = list(events)

    def history(self):
        return list(self._events)

    def clear(self):
        self._events = []


class _FakeResult:
    def __init__(self, status, summary="", error=""):
        self.status = status
        self.summary = summary
        self.error = error


class _FakeRunner:
    def __init__(self, result):
        self.busy = False
        self.result = result


def _result_app(tmp_path, result, events=()):
    from agent.config import AgentConfig
    from agent.tui import TuiApp

    app = TuiApp(AgentConfig(workspace=tmp_path))
    app._hub = _FakeHub(events)
    app._last_event_count = 0
    app._runner = _FakeRunner(result)
    return app


def _visible_text(app):
    return "\n".join(m.get("content", "") for m in app.messages)


@pytest.mark.parametrize(
    ("status", "summary", "error"),
    [
        ("fatal", "Ollama is unavailable.", "cannot reach localhost: refused"),
        ("fatal", "Ollama request timed out.", "timed out after 600s"),
        ("fatal", "Model missing.", "Ollama HTTP 404: not found"),
        ("malformed", "Bad reply.", "expected tool call"),
        ("max_iterations", "Stopped after 50 iterations.", ""),
        ("interrupted", "Interrupted.", ""),
        ("failed", "Task failed.", "boom"),
    ],
)
def test_terminal_failures_are_readable_never_success(
    tmp_path, status, summary, error
):
    app = _result_app(tmp_path, _FakeResult(status, summary, error))
    app._poll_runner()
    text = _visible_text(app)
    assert "Traceback (most recent call last)" not in text
    assert "qwen3-coder:30b" in text
    assert "ollama" in text.lower()
    assert app.status_msg == "Failed — ready"
    assert not any(
        m.get("role") == "assistant" and "Completed" in m.get("content", "")
        for m in app.messages
    )


def test_terminal_cancelled_stays_cancelled(tmp_path):
    app = _result_app(tmp_path, _FakeResult("cancelled", "Stopped.", ""))
    app._poll_runner()
    assert _visible_text(app) == "Cancelled."
    assert app.status_msg == "Cancelled — ready"


def test_terminal_completed_still_succeeds(tmp_path):
    app = _result_app(tmp_path, _FakeResult("completed", "All done.", ""))
    app._poll_runner()
    assert any(
        m.get("role") == "assistant" and "All done." in m.get("content", "")
        for m in app.messages
    )
    assert app.status_msg == "Completed — ready"


def test_terminal_failure_clears_generating_state(tmp_path):
    app = _result_app(
        tmp_path,
        _FakeResult("fatal", "Ollama is unavailable.", "refused"),
        events=[_FakeEvent("model_started", "thinking")],
    )
    app._poll_runner()
    assert "Thinking" not in app.status_msg
    assert "Processing" not in app.status_msg
    assert app._runner is None


def test_mid_run_error_updates_status(tmp_path):
    app = _result_app(
        tmp_path,
        _FakeResult("completed", "Recovered and done.", ""),
        events=[_FakeEvent("agent_error", "boom")],
    )
    # Slow path drains the error event first: status must not linger.
    app._hub = _FakeHub([_FakeEvent("agent_error", "boom")])
    app._last_event_count = 0
    app._runner = None  # error event only, no terminal result yet
    app._poll_runner()
    assert app.status_msg == "Failed — ready"
    assert "boom" in _visible_text(app)


def test_start_failure_names_provider_and_model(tmp_path):
    from agent.config import AgentConfig
    from agent.tui import TuiApp

    app = TuiApp(AgentConfig(workspace=tmp_path))
    app._add_system = app._add_system  # keep real recording
    text = app._startup_error("client", OSError("connection refused" * 50))
    assert "qwen3-coder:30b" in text
    assert "ollama" in text
    assert len(text) < 600
    assert "Traceback" not in text


def test_picker_still_usable_after_error(tmp_path, monkeypatch):
    from agent.config import AgentConfig
    from agent.tui import TuiApp

    monkeypatch.setattr(
        "agent.models.providers.list_all_providers_with_models",
        lambda timeout=2, use_cache=True: {"ollama": []},
    )
    app = TuiApp(
        AgentConfig(
            workspace=tmp_path, provider="ollama", model="qwen3-coder:30b"
        )
    )

    class Dummy:
        def getmaxyx(self):
            return (24, 80)

    app._handle_slash(Dummy(), "/models")  # empty state, no crash
    assert "/connect" in app.status_msg
    app._finish_task_result(_FakeResult("completed", "ok", ""))
    assert app.status_msg == "Completed — ready"  # session usable
