"""Tests for the structured agent event contract (agent.events)."""

from __future__ import annotations

import json

from agent.events import (
    EVENT_TYPES,
    AgentEvent,
    emit_command_output,
    emit_model_completed,
    emit_model_started,
    emit_retry,
    emit_status,
    emit_task_blocked,
    emit_task_created,
    emit_task_ready,
    emit_task_verified,
    emit_tool_completed,
    emit_verification_started,
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


def test_task_lifecycle_events_are_declared():
    for t in (
        "task_created",
        "task_ready",
        "task_blocked",
        "verification_started",
        "retry",
    ):
        assert t in EVENT_TYPES


def test_task_created_event_carries_deps_and_files():
    out = []
    emit_task_created(
        out.append,
        "t1",
        "Implement parser",
        depends_on=("t0",),
        n_files=3,
    )
    ev = out[0]
    assert ev.type == "task_created"
    assert ev.status == "t1"
    assert ev.message == "Implement parser"
    data = ev.to_dict()
    assert data["summary"] == "1 dep(s), 3 file(s)"


def test_task_ready_and_blocked_events():
    out = []
    emit_task_ready(out.append, "t1", "Implement parser")
    emit_task_blocked(out.append, "t2", "Fix tests", reason="dependency not satisfied")
    assert [e.type for e in out] == ["task_ready", "task_blocked"]
    assert out[0].status == "t1"
    assert out[1].error == "dependency not satisfied"


def test_retry_event_carries_attempt_and_retries_left():
    out = []
    emit_retry(out.append, task_id="t1", attempt=1, retries_left=2, reason="tests fail")
    ev = out[0]
    assert ev.type == "retry"
    assert ev.attempt == 1
    assert ev.retries_left == 2
    data = ev.to_dict()
    assert data["attempt"] == 1
    assert data["retries_left"] == 2


def test_verification_started_event_carries_attempt():
    out = []
    emit_verification_started(out.append, "t1", attempt=2)
    ev = out[0]
    assert ev.type == "verification_started"
    assert ev.status == "t1"
    assert ev.attempt == 2


def test_task_verified_carries_structured_retry_fields():
    out = []
    emit_task_verified(
        out.append, "t1", ok=False, summary="boom",
        attempt=3, retries_left=0,
    )
    ev = out[0]
    assert ev.type == "task_verified"
    assert ev.ok is False
    assert ev.attempt == 3
    assert ev.retries_left == 0
    data = ev.to_dict()
    assert data["attempt"] == 3
    assert data["retries_left"] == 0
    json.dumps(data)  # serializable


def test_task_verified_optional_retry_fields_dropped_when_absent():
    out = []
    emit_task_verified(out.append, "t1", ok=True, summary="ok")
    ev = out[0]
    data = ev.to_dict()
    assert "attempt" not in data
    assert "retries_left" not in data