"""Tests for export_handover_state method."""

import json
import pytest

from agent.core.loop import AgentLoop
from agent.config import load_config
from agent.workspace import Workspace
from agent.models.client import OllamaClient

def make_loop(tmp_path):
    config = load_config(workspace=tmp_path)
    client = OllamaClient(base_url=config.ollama_base_url, model=config.model, request_timeout=config.request_timeout)
    workspace = Workspace(tmp_path)
    return AgentLoop(config, client, workspace)

def test_export_handover_state_serializable(tmp_path):
    loop = make_loop(tmp_path)
    state = loop.export_handover_state()

    json_str = json.dumps(state)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert "task" in parsed
    assert "plan" in parsed
    assert "completed_actions" in parsed
    assert "observations" in parsed
    assert "partial_results" in parsed
    assert "context_index_ref" in parsed
    assert "experience_tags" in parsed
    assert "workspace" in parsed
    assert "timestamp" in parsed
    assert isinstance(parsed["timestamp"], str)

def test_export_handover_state_returns_dict(tmp_path):
    loop = make_loop(tmp_path)
    state = loop.export_handover_state()
    assert isinstance(state, dict)
    json_str = json.dumps(state)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
