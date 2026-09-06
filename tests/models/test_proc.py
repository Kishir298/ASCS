"""Tests for Ollama server shutdown on CLI exit (agent.models.proc)."""

from __future__ import annotations

from agent.config import AgentConfig
from agent.models.proc import (
    maybe_stop_ollama_on_exit,
    stop_ollama_server,
    stop_on_exit_enabled,
)


class _Recorder:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[0] in self.fail_on:
            raise OSError(f"no {argv[0]}")
        return None


def test_stop_unloads_model_then_kills_server(tmp_path):
    rec = _Recorder()
    assert stop_ollama_server("qwen3-coder:30b", run=rec) is True
    assert rec.calls[0] == ["ollama", "stop", "qwen3-coder:30b"]
    assert rec.calls[1][0] in ("taskkill", "pkill")


def test_stop_never_raises(tmp_path):
    rec = _Recorder(fail_on=("ollama",))
    assert stop_ollama_server("qwen3-coder:30b", run=rec) is False


def test_opt_out_env_disables_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_STOP_OLLAMA_ON_EXIT", "false")
    assert stop_on_exit_enabled() is False
    monkeypatch.setenv("AGENT_STOP_OLLAMA_ON_EXIT", "1")
    assert stop_on_exit_enabled() is True


def test_maybe_stop_respects_config_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_STOP_OLLAMA_ON_EXIT", raising=False)
    rec = _Recorder()
    import agent.models.proc as proc_mod

    monkeypatch.setattr(proc_mod.subprocess, "run", rec)
    config = AgentConfig(workspace=tmp_path, stop_ollama_on_exit=False)
    maybe_stop_ollama_on_exit(config)
    assert rec.calls == []


def test_maybe_stop_runs_when_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_STOP_OLLAMA_ON_EXIT", raising=False)
    rec = _Recorder()
    import agent.models.proc as proc_mod

    monkeypatch.setattr(proc_mod.subprocess, "run", rec)
    config = AgentConfig(workspace=tmp_path, stop_ollama_on_exit=True)
    maybe_stop_ollama_on_exit(config)
    assert rec.calls[0] == ["ollama", "stop", config.model]


def test_config_defaults_lock_primary_and_fallback(tmp_path):
    from agent.config import DEFAULT_MODEL, FALLBACK_MODEL

    assert DEFAULT_MODEL == "qwen3-coder:30b"
    assert FALLBACK_MODEL == "qwen2.5-coder:14b"
    config = AgentConfig(workspace=tmp_path)
    assert config.model == "qwen3-coder:30b"
    assert config.fallback_model == "qwen2.5-coder:14b"
    assert config.stop_ollama_on_exit is True


def test_entry_diagnostics_never_stop_server(tmp_path, monkeypatch):
    """--check/--doctor/--list-models are read-only: no shutdown."""
    import agent.terminal.entry as entry_mod

    monkeypatch.setattr(entry_mod, "_cmd_check", lambda client: 0)
    stopped = []
    monkeypatch.setattr(
        proc_mod_for_test(), "maybe_stop_ollama_on_exit", stopped.append
    )
    rc = entry_mod.main(["--check"])
    assert rc == 0
    assert stopped == []


def proc_mod_for_test():
    import agent.models.proc as proc_mod

    return proc_mod


def test_entry_tui_exit_stops_server(tmp_path, monkeypatch):
    import agent.terminal.entry as entry_mod

    monkeypatch.setattr(entry_mod, "_cmd_tui", lambda config, client: 0)
    stopped = []
    monkeypatch.setattr(
        proc_mod_for_test(), "maybe_stop_ollama_on_exit", stopped.append
    )
    monkeypatch.delenv("AGENT_STOP_OLLAMA_ON_EXIT", raising=False)
    rc = entry_mod.main(["--tui", "--workspace", str(tmp_path)])
    assert rc == 0
    assert len(stopped) == 1
