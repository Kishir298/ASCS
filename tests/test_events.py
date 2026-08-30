"""Tests for the structured agent event contract (agent.events)."""

from __future__ import annotations

import json

from agent.events import (
    EVENT_TYPES,
    AgentEvent,
    emit_command_output,
    emit_model_completed,
    emit_model_started,
    emit_status,
    emit_tool_completed,
    to_event_dict,
)


def _collect():
    events = []
    return events.append, events


def test_to_dict_drops_none_fields():
    ev = AgentEvent(type="status", status="RUNNING")
    data = ev.to_dict()
    assert data["type"] == "status"
    assert data["status"] == "RUNNING"
    assert data["message"] == ""  # default empty string is a value, not None
    assert "error" not in data  # None fields are dropped
    assert "output" not in data


def test_event_json_roundtrip():
    ev = AgentEvent(
        type="tool_completed",
        tool="run_command",
        command="pytest",
        exit_code=0,
        ok=True,
        elapsed=1.25,
    )
    parsed = json.loads(ev.to_json())
    assert parsed["type"] == "tool_completed"
    assert parsed["tool"] == "run_command"
    assert parsed["exit_code"] == 0
    assert parsed["elapsed"] == 1.25
    assert "timestamp" in parsed


def test_to_event_dict_is_json_serializable():
    ev = AgentEvent(type="activity", message="hello")
    json.dumps(to_event_dict(ev))  # must not raise


def test_model_lifecycle_events():
    out = []
    emit_model_started(out.append, "thinking")
    emit_model_completed(out.append, "responded", elapsed=0.5)
    types = [e.type for e in out]
    assert types == ["model_started", "model_completed"]
    assert out[1].elapsed == 0.5


def test_command_output_event_carries_payload():
    out = []
    emit_command_output(
        out.append,
        "pytest",
        "2 passed",
        exit_code=0,
        ok=True,
    )
    ev = out[0]
    assert ev.type == "command_output"
    assert ev.command == "pytest"
    assert ev.output == "2 passed"
    assert ev.exit_code == 0
    assert ev.ok is True


def test_tool_completed_event_ok_flag():
    out = []
    emit_tool_completed(out.append, "apply_patch", ok=False, target="a.py", elapsed=1.0)
    ev = out[0]
    assert ev.ok is False
    assert ev.target == "a.py"
    data = ev.to_dict()
    assert data["ok"] is False


def test_event_types_are_covered_by_emitters():
    # Every declared event type must have a configurable field set that the
    # loop can produce; these are the canonical lifecycle events.
    required = {
        "agent_started",
        "status",
        "mode_changed",
        "model_started",
        "model_completed",
        "tool_started",
        "tool_completed",
        "command_started",
        "command_output",
        "command_completed",
        "test_started",
        "test_completed",
        "agent_error",
        "agent_stopped",
        "agent_completed",
    }
    assert required <= set(EVENT_TYPES)


def test_status_event_roundtrip():
    events = []
    emit_status(events.append, "PLANNING", "model is analysing")
    ev = events[0]
    assert ev.status == "PLANNING"
    data = json.loads(ev.to_json())
    assert data["status"] == "PLANNING"
    assert data["message"] == "model is analysing"