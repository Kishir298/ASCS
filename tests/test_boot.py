"""Tests for the staged startup sequence (agent.boot)."""

from __future__ import annotations

import pytest

from agent.boot import boot, boot_error_message, print_boot
from agent.ollama import OllamaError


class FakeOllamaClient:
    """Response-scripted stand-in for the real Ollama client."""

    def __init__(self, report, error=None):
        self.report = report
        self.error = error
        self.ensure_ready_calls = 0
        self.base_url = "http://fake:11434"
        self.model = report.get("model", "fake-model")
        self.keep_alive = None
        self.request_timeout = 60

    def ensure_ready(self, *, check_timeout=8, warm_timeout=120, prewarm=False):
        self.ensure_ready_calls += 1
        if self.error:
            raise self.error
        return self.report


def _patch(monkeypatch):
    monkeypatch.setattr(
        "agent.boot.OllamaClient",
        lambda base_url, model, request_timeout, keep_alive: FakeOllamaClient(
            {
                "reachable": True,
                "version": "0.5.0",
                "model": model or "fake-model",
                "available": True,
                "installed": ["fake-model"],
                "warmed": False,
            }
        ),
    )


def test_boot_success_stage_ordering(monkeypatch, tmp_path):
    _patch(monkeypatch)
    phases = []
    report = boot(
        workspace_path=str(tmp_path),
        model="fake-model",
        progress=lambda phase, message, elapsed: phases.append(phase),
    )
    assert report.ok
    assert report.error == ""
    assert report.config is not None
    assert report.workspace is not None
    # config -> pyenv -> workspace -> ollama -> model -> tools -> env
    assert phases[:5] == ["config", "pyenv", "workspace", "ollama", "model"]
    assert "tools" in phases
    assert "env" in phases
    stage_phases = [s["phase"] for s in report.stages]
    assert stage_phases == phases


def test_boot_prewarm_calls_ensure_ready_twice(monkeypatch, tmp_path):
    calls = {}

    class Counting(FakeOllamaClient):
        def ensure_ready(self, *, check_timeout=8, warm_timeout=120, prewarm=False):
            key = "warm" if prewarm else "check"
            calls[key] = calls.get(key, 0) + 1
            return super().ensure_ready(
                check_timeout=check_timeout, warm_timeout=warm_timeout, prewarm=prewarm
            )

    monkeypatch.setattr(
        "agent.boot.OllamaClient",
        lambda base_url, model, request_timeout, keep_alive: Counting(
            {"reachable": True, "version": "0.5.0", "model": "m", "available": True,
             "installed": ["m"], "warmed": False}
        ),
    )
    report = boot(workspace_path=str(tmp_path), prewarm=True, model="m")
    assert report.ok
    assert calls.get("check") == 1
    assert calls.get("warm") == 1


def test_boot_ollama_unreachable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent.boot.OllamaClient",
        lambda base_url, model, request_timeout, keep_alive: FakeOllamaClient(
            {"available": True}, error=OllamaError("down")
        ),
    )
    report = boot(workspace_path=str(tmp_path), prewarm=False)
    assert not report.ok
    assert report.error_phase == "ollama"
    assert "ollama" in boot_error_message(report).lower()


def test_boot_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent.boot.OllamaClient",
        lambda base_url, model, request_timeout, keep_alive: FakeOllamaClient(
            {"reachable": True, "version": "0.5.0", "model": model,
             "available": False, "installed": ["other-model"], "warmed": False}
        ),
    )
    report = boot(workspace_path=str(tmp_path), model="missing-model", prewarm=False)
    assert not report.ok
    assert report.error_phase == "model"
    assert "not installed" in report.error
    assert "missing-model" in report.error


def test_boot_workspace_missing(monkeypatch, tmp_path):
    _patch(monkeypatch)
    report = boot(workspace_path=str(tmp_path / "does-not-exist"), prewarm=False)
    assert not report.ok
    assert report.error_phase == "workspace"


def test_boot_invalid_config(monkeypatch, tmp_path):
    _patch(monkeypatch)
    report = boot(workspace_path=str(tmp_path), mode="BOGUS", prewarm=False)
    assert not report.ok
    assert report.error_phase == "config"


def test_boot_report_error_message_default():
    from agent.boot import BootReport

    report = BootReport()
    assert "unknown" in boot_error_message(report)


def test_print_boot_does_not_raise(monkeypatch, tmp_path, capsys):
    _patch(monkeypatch)
    report = boot(workspace_path=str(tmp_path), model="fake-model")
    print_boot(report)
    out = capsys.readouterr().out
    assert "A.S.C.S." in out
    assert "ready" in out.lower()