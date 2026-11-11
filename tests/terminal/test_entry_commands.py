"""Regression tests for the installed ``ascs`` subcommands."""

from __future__ import annotations

import json

from agent.terminal import entry as entry_mod


def test_setup_creates_private_state_without_network(tmp_path, monkeypatch, capsys):
    state = tmp_path / "state" / "tui_state.json"
    monkeypatch.setenv("AGENT_TUI_STATE_PATH", str(state))
    assert entry_mod.main(["setup"]) == 0
    assert state.is_file()
    data = json.loads(state.read_text(encoding="utf-8"))
    assert data["mode"] == "AUTO"
    assert data["provider"]
    assert data["model"]
    assert "setup complete" in capsys.readouterr().out.lower()


def test_setup_preserves_existing_state(tmp_path, monkeypatch):
    state = tmp_path / "tui_state.json"
    state.write_text(json.dumps({"provider": "custom", "model": "mine"}), encoding="utf-8")
    monkeypatch.setenv("AGENT_TUI_STATE_PATH", str(state))
    assert entry_mod.main(["setup"]) == 0
    data = json.loads(state.read_text(encoding="utf-8"))
    assert data["provider"] == "custom"
    assert data["model"] == "mine"
    assert data["mode"] == "AUTO"


def test_doctor_subcommand_runs_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_TUI_STATE_PATH", str(tmp_path / "tui_state.json"))
    rc = entry_mod.main(["doctor", str(tmp_path)])
    out = capsys.readouterr().out
    assert "A.S.C.S. doctor" in out
    assert rc in (0, 1)
