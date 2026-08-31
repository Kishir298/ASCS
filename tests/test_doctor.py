"""Tests for the A.S.C.S. diagnostics (agent.doctor / risa --doctor)."""

from __future__ import annotations

import pytest

from agent import doctor as doctor_mod
from agent.doctor import doctor, print_doctor
from agent.main import build_parser


class FakeClient:
    """Scripted stand-in for OllamaClient used by the doctor checks."""

    def __init__(self, *, base_url="http://fake:11434", model="m", request_timeout=600,
                 keep_alive=None, reachable=True, installed=None, verify_error=None):
        self.base_url = base_url
        self.model = model
        self.request_timeout = request_timeout
        self.keep_alive = keep_alive
        self.reachable = reachable
        self.installed = installed if installed is not None else ["m"]
        self.verify_error = verify_error

    def check_connectivity(self, timeout=5):
        if self.verify_error:
            raise RuntimeError(self.verify_error)
        return self.reachable

    def list_models(self, timeout=10):
        if self.verify_error:
            raise RuntimeError(self.verify_error)
        return list(self.installed)


@pytest.fixture
def patch_ollama(monkeypatch):
    def _patch(**kwargs):
        fake = FakeClient(**kwargs)
        monkeypatch.setattr(
            "agent.doctor.OllamaClient",
            lambda base_url, model, request_timeout, keep_alive=None: fake,
        )
        return fake

    return _patch


def test_doctor_happy_path(tmp_path, patch_ollama):
    patch_ollama(model="m", installed=["m"])
    report = doctor(workspace=tmp_path, model="m")
    assert report.ok
    names = {r.name for r in report.results}
    assert {"python", "install", "config", "workspace", "ollama", "model",
            "tools", "context", "project", "git", "tests"} <= names
    # No failures anywhere; context and git may legitimately WARN.
    assert report.failed == []
    assert next(r for r in report.results if r.name == "ollama").status == "PASS"
    assert next(r for r in report.results if r.name == "model").status == "PASS"


def test_doctor_missing_workspace(tmp_path, patch_ollama):
    patch_ollama()
    missing = tmp_path / "nope"
    report = doctor(workspace=missing)
    assert not report.ok
    ws = next(r for r in report.results if r.name == "workspace")
    assert ws.status == "FAIL"
    assert "does not exist" in ws.message


def test_doctor_config_error(tmp_path, patch_ollama):
    patch_ollama()
    report = doctor(workspace=tmp_path, mode="BOGUS")
    assert not report.ok
    cfg = report.results[0]
    assert cfg.name == "config"
    assert cfg.status == "FAIL"


def test_doctor_ollama_unreachable(tmp_path, patch_ollama):
    patch_ollama(reachable=True, verify_error="connection refused")
    report = doctor(workspace=tmp_path)
    assert not report.ok
    oll = next(r for r in report.results if r.name == "ollama")
    assert oll.status == "FAIL"
    assert "ollama serve" in oll.message.lower()


def test_doctor_model_missing(tmp_path, patch_ollama):
    patch_ollama(model="missing", installed=["other"])
    report = doctor(workspace=tmp_path, model="missing")
    assert not report.ok
    model = next(r for r in report.results if r.name == "model")
    assert model.status == "FAIL"
    assert "ollama pull" in model.message.lower()


def test_doctor_context_warn_without_index(tmp_path, patch_ollama):
    patch_ollama()
    report = doctor(workspace=tmp_path)
    ctx = next(r for r in report.results if r.name == "context")
    assert ctx.status == "WARN"
    assert "will build" in ctx.message.lower()


def test_doctor_context_pass_with_index(tmp_path, patch_ollama):
    patch_ollama()
    from agent.context import ProjectIndex

    idx = ProjectIndex(tmp_path)
    idx.save()
    report = doctor(workspace=tmp_path)
    ctx = next(r for r in report.results if r.name == "context")
    assert ctx.status == "PASS"
    assert "records" in ctx.message.lower()


def test_print_doctor_renders_statuses(tmp_path, patch_ollama, capsys):
    patch_ollama()
    report = doctor(workspace=tmp_path)
    print_doctor(report)
    out = capsys.readouterr().out
    assert "A.S.C.S. doctor" in out
    assert "[PASS] python" in out


def test_parser_has_doctor_flag():
    ns = build_parser().parse_args(["--doctor"])
    assert ns.doctor is True


def test_cmd_doctor_returns_zero_when_ok(tmp_path, monkeypatch):
    class _Fake:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr("agent.doctor.OllamaClient", _Fake)
    monkeypatch.setattr(
        "agent.doctor._check_context_index",
        lambda cfg: doctor_mod.CheckResult("context", "PASS", "ok"),
    )
    monkeypatch.setattr(
        "agent.doctor._check_ollama",
        lambda cfg: doctor_mod.CheckResult("ollama", "PASS", "ok"),
    )
    monkeypatch.setattr(
        "agent.doctor._check_model",
        lambda cfg: doctor_mod.CheckResult("model", "PASS", "ok"),
    )
    from agent.main import main

    rc = main(["--doctor", "--workspace", str(tmp_path)])
    assert rc == 0


def test_cmd_doctor_returns_nonzero_when_failing(tmp_path, monkeypatch):
    class _Fake:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr("agent.doctor.OllamaClient", _Fake)
    monkeypatch.setattr(
        "agent.doctor._check_ollama",
        lambda cfg: doctor_mod.CheckResult("ollama", "FAIL", "down"),
    )
    monkeypatch.setattr(
        "agent.doctor._check_model",
        lambda cfg: doctor_mod.CheckResult("model", "FAIL", "missing"),
    )
    from agent.main import main

    rc = main(["--doctor", "--workspace", str(tmp_path)])
    assert rc == 1