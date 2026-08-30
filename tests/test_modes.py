"""Tests for PLAN / BUILD / AUTO mode semantics in the agent loop."""

from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.state import COMPLETE, EXECUTING, PLANNING

from test_loop import make_loop, tool_call


def test_plan_mode_never_modifies_files(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("list_directory", {}),
            tool_call(
                "set_plan",
                {"goal": "add hello", "plan": ["inspect", "write", "test"]},
            ),
            tool_call("write_file", {"path": "x.py", "content": "x=1"}),
            json.dumps({"done": True, "summary": "planned"}),
        ],
        config_overrides={"mode": "PLAN"},
    )
    result = loop.run("plan a change")
    assert result.is_complete
    # The write_file attempt is not enabled in PLAN mode -> never executed.
    assert not (tmp_path / "x.py").exists()
    assert result.plan is not None
    assert len(result.plan.steps) == 3
    assert result.plan.goal == "add hello"


def test_plan_mode_allows_only_inspection_tools(tmp_path):
    cfg = AgentConfig(workspace=tmp_path, mode="PLAN")
    tools = set(cfg.effective_tools)
    assert tools == {
        "list_directory",
        "read_file",
        "search_files",
        "inspect_environment",
        "git_status",
        "git_diff",
        "set_plan",
    }


def test_build_mode_executes_and_records_plan(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("list_directory", {}),
            tool_call("set_plan", {"plan": ["create file"]}),
            tool_call("write_file", {"path": "a.txt", "content": "hello"}),
            json.dumps({"done": True, "summary": "built"}),
        ],
        config_overrides={"mode": "BUILD"},
    )
    result = loop.run("make a.txt")
    assert result.is_complete
    assert (tmp_path / "a.txt").read_text() == "hello"
    assert result.plan is not None
    assert result.plan.steps == ["create file"]


def test_auto_mode_runs_fully_autonomously(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("write_file", {"path": "f.py", "content": "print(1)\n"}),
            json.dumps({"done": True, "summary": "auto done"}),
        ],
        config_overrides={"mode": "AUTO"},
    )
    result = loop.run("task")
    assert result.is_complete
    assert (tmp_path / "f.py").exists()


def test_safe_mode_is_approval_overlay(tmp_path):
    approver_calls = []
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("write_file", {"path": "x.py", "content": "x=1"}),
            json.dumps({"done": True, "summary": "done"}),
        ],
        config_overrides={"mode": "SAFE"},
        approver=lambda d: approver_calls.append(d) or True,
    )
    result = loop.run("task")
    assert result.is_complete
    assert approver_calls  # approval asked
    assert (tmp_path / "x.py").exists()


def test_plan_mode_lifecycle_states(tmp_path):
    tracker_states = []
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("list_directory", {}),
            tool_call("set_plan", {"plan": ["step one"]}),
            json.dumps({"done": True, "summary": "ok"}),
        ],
        config_overrides={"mode": "PLAN"},
    )
    loop.tracker.on_transition(lambda s, p: tracker_states.append(s))
    loop.run("plan it")
    assert PLANNING in tracker_states
    assert COMPLETE in tracker_states


def test_build_enters_executing_on_modify(tmp_path):
    states = []
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("write_file", {"path": "m.py", "content": "x=1"}),
            json.dumps({"done": True, "summary": "ok"}),
        ],
        config_overrides={"mode": "BUILD"},
    )
    loop.tracker.on_transition(lambda s, p: states.append(s))
    loop.run("task")
    assert EXECUTING in states
    assert COMPLETE in states