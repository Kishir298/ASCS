"""Tests for configuration resolution and validation."""

from __future__ import annotations

import pytest

from agent.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    AgentConfig,
    load_config,
)


def test_defaults(monkeypatch, tmp_path):
    # Clear any ambient env that could affect defaults.
    for name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "AGENT_MAX_ITERATIONS",
        "AGENT_MODE",
        "AGENT_COMMAND_TIMEOUT",
        "AGENT_MAX_OUTPUT_CHARS",
        "AGENT_CONTEXT_BUDGET_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = load_config(workspace=str(tmp_path))
    assert cfg.ollama_base_url == DEFAULT_OLLAMA_BASE_URL
    assert cfg.model == DEFAULT_MODEL
    assert cfg.max_iterations == DEFAULT_MAX_ITERATIONS
    assert cfg.mode == "AUTO"
    assert not cfg.is_safe_mode


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:9999")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-coder:6.7b")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "7")
    monkeypatch.setenv("AGENT_MODE", "SAFE")
    cfg = load_config(workspace=str(tmp_path))
    assert cfg.ollama_base_url == "http://example:9999"
    assert cfg.model == "deepseek-coder:6.7b"
    assert cfg.max_iterations == 7
    assert cfg.mode == "SAFE"
    assert cfg.is_safe_mode


def test_explicit_overrides_beat_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    cfg = load_config(workspace=str(tmp_path), model="from-cli")
    assert cfg.model == "from-cli"


def test_workspace_does_not_exist(tmp_path):
    with pytest.raises(ValueError):
        load_config(workspace=str(tmp_path / "nope"))


def test_workspace_is_not_a_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        load_config(workspace=str(f))


def test_env_int_invalid(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "abc")
    with pytest.raises(ValueError):
        load_config(workspace=str(tmp_path))


def test_env_int_nonpositive(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "0")
    with pytest.raises(ValueError):
        load_config(workspace=str(tmp_path))


def test_invalid_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODE", "BANANA")
    with pytest.raises(ValueError):
        load_config(workspace=str(tmp_path))


def test_tools_tuple_preset(tmp_path):
    cfg = load_config(workspace=str(tmp_path))
    assert cfg.tools == (
        "list_directory",
        "read_file",
        "search_files",
        "write_file",
        "apply_patch",
        "delete_file",
        "move_file",
        "copy_file",
        "run_command",
        "inspect_environment",
        "git_status",
        "git_diff",
        "set_plan",
    )


def test_config_is_frozen():
    cfg = AgentConfig()
    with pytest.raises(AttributeError):
        cfg.model = "other"
