"""Tests for the autonomous agent loop using a fake (scripted) model client.

A real Ollama server is intentionally NOT required.
"""

from __future__ import annotations

import json

import pytest

from agent.config import AgentConfig
from agent.loop import AgentLoop, run_agent
from agent.ollama import (
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaModelNotFoundError,
)
from agent.workspace import Workspace


class FakeClient:
    """Returns a pre-scripted list of chat responses in order."""

    def __init__(self, responses, model="fake-model"):
        self.responses = list(responses)
        self.index = 0
        self.model = model
        self.calls = []

    def chat(self, messages, *, format="json", options=None, timeout=None):
        self.calls.append(messages)
        if self.index < len(self.responses):
            item = self.responses[self.index]
            self.index += 1
            if isinstance(item, BaseException):
                raise item
            return item
        raise AssertionError("FakeClient exhausted scripted responses")


def make_loop(tmp_path, responses, config_overrides=None, approver=None):
    cfg_kwargs = {"workspace": tmp_path, "mode": "AUTO"}
    if config_overrides:
        cfg_kwargs.update(config_overrides)
    config = AgentConfig(**cfg_kwargs)
    client = FakeClient(responses, model=config.model)
    ws = Workspace(tmp_path)
    loop = AgentLoop(config, client, ws, approver=approver, log=lambda m: None)
    return loop, client


def tool_call(tool, arguments, comment="step"):
    return json.dumps({"comment": comment, "tool": tool, "arguments": arguments})


def test_completes_task_with_tools(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("write_file", {"path": "hello.py", "content": "print('hi')\n"}),
            tool_call("run_command", {"command": "python hello.py"}),
            json.dumps({"done": True, "summary": "created and ran hello.py"}),
        ],
    )
    result = loop.run("Make hello.py and run it")
    assert result.is_complete
    assert result.status == "completed"
    assert "hello.py" in result.summary
    assert (tmp_path / "hello.py").exists()


def test_reaches_max_iterations(tmp_path):
    # The model never returns "done": the loop must stop at the limit.
    loop, _ = make_loop(
        tmp_path,
        [tool_call("git_status", {})] * 5,
        config_overrides={"max_iterations": 3},
    )
    result = loop.run("keep going")
    assert result.status == "max_iterations"
    assert result.iterations == 3


def test_malformed_response_hits_limit(tmp_path):
    # Every reply is incomprehensible; loop must stop after the retry limit.
    loop, _ = make_loop(
        tmp_path,
        ["this is not json at all"] * 10,
        config_overrides={"malformed_retry_limit": 2},
    )
    result = loop.run("anything")
    assert result.status == "malformed"


def test_repeated_identical_failing_call_stops(tmp_path):
    # First call fails, then the exact same call repeats -> fatal loop guard.
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("read_file", {"path": "missing.txt"}),
            tool_call("read_file", {"path": "missing.txt"}),
            tool_call("read_file", {"path": "missing.txt"}),
        ],
    )
    result = loop.run("debug")
    assert result.status == "fatal"
    assert "repeated" in result.summary.lower()


def test_safe_mode_approver_declines(tmp_path):
    approver_calls = []

    def approver(desc):
        approver_calls.append(desc)
        return False  # decline everything

    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("write_file", {"path": "x.py", "content": "x=1"}),
            json.dumps({"done": True, "summary": "finished"}),
        ],
        config_overrides={"mode": "SAFE"},
        approver=approver,
    )
    result = loop.run("task")
    assert result.status == "completed"
    assert "write_file" in " ".join(approver_calls)
    assert not (tmp_path / "x.py").exists()  # declined -> not executed


def test_safe_mode_approver_approves(tmp_path):
    def approver(desc):
        return True

    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("write_file", {"path": "x.py", "content": "x=1"}),
            json.dumps({"done": True, "summary": "done"}),
        ],
        config_overrides={"mode": "SAFE"},
        approver=approver,
    )
    result = loop.run("task")
    assert result.status == "completed"
    assert (tmp_path / "x.py").exists()


def test_interrupt_returns_interrupted(tmp_path):
    loop, _ = make_loop(tmp_path, [KeyboardInterrupt()])
    result = loop.run("anything")
    assert result.status == "interrupted"


def test_ollama_connection_error_is_fatal(tmp_path):
    loop, _ = make_loop(tmp_path, [OllamaConnectionError("offline")])
    result = loop.run("anything")
    assert result.status == "fatal"


def test_ollama_model_not_found_is_fatal(tmp_path):
    loop, _ = make_loop(tmp_path, [OllamaModelNotFoundError(404, "missing")])
    result = loop.run("anything")
    assert result.status == "fatal"


def test_generic_ollama_http_error_is_fatal_not_crash(tmp_path):
    # A non-404 HTTP error must yield a clean fatal result, not an exception.
    loop, _ = make_loop(tmp_path, [OllamaHTTPError(500, "boom")])
    result = loop.run("anything")
    assert result.status == "fatal"
    assert result.error


def test_disabled_tool_gets_feedback_not_crash(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("git_status", {}),
            json.dumps({"done": True, "summary": "done"}),
        ],
        config_overrides={"tools": ("read_file",)},
    )
    result = loop.run("task")
    assert result.status == "completed"


def test_comment_is_logged_to_steps(tmp_path):
    steps = []

    def log(m):
        steps.append(m)

    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    client = FakeClient(
        [json.dumps({"comment": "annotated", "tool": "git_status", "arguments": {}}),
         json.dumps({"done": True, "summary": "s"})],
        model=config.model,
    )
    loop = AgentLoop(config, client, Workspace(tmp_path), log=log)
    result = loop.run("task")
    assert result.status == "completed"
    assert any("annotated" in s for s in steps)


def test_run_agent_helper(tmp_path):
    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    client = FakeClient(
        [json.dumps({"done": True, "summary": "already done"})], model=config.model
    )
    result = run_agent(config, client, "task")
    assert result.status == "completed"


def test_plan_captured_from_set_plan(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call(
                "set_plan",
                {"goal": "refactor", "plan": ["inspect", "edit", "test"]},
            ),
            json.dumps({"done": True, "summary": "planned"}),
        ],
    )
    result = loop.run("refactor")
    assert result.is_complete
    assert result.plan is not None
    assert result.plan.goal == "refactor"
    assert result.plan.steps == ["inspect", "edit", "test"]


def test_state_machine_reaches_complete(tmp_path):
    loop, _ = make_loop(tmp_path, [json.dumps({"done": True, "summary": "ok"})])
    result = loop.run("task")
    assert result.state == "complete"
    assert result.status == "completed"
    assert loop.tracker.state == "complete"


def test_cancelled_state_never_success(tmp_path):
    loop, _ = make_loop(tmp_path, [KeyboardInterrupt()])
    result = loop.run("task")
    assert result.status == "interrupted"
    assert result.state == "cancelled"
    assert not result.is_complete


def test_max_iterations_maps_to_timeout_state(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [tool_call("git_status", {})] * 5,
        config_overrides={"max_iterations": 2},
    )
    result = loop.run("keep going")
    assert result.state == "timeout"
    assert result.status == "max_iterations"


def test_fatal_error_maps_to_failed_state(tmp_path):
    loop, _ = make_loop(tmp_path, [OllamaConnectionError("offline")])
    result = loop.run("anything")
    assert result.state == "failed"
    assert result.status == "fatal"


def test_should_stop_true_aborts_with_cancelled(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [json.dumps({"done": True, "summary": "should not run"})],
        config_overrides={"max_iterations": 5},
    )
    loop.should_stop = lambda: True
    result = loop.run("task")
    assert result.status == "cancelled"
    assert result.state == "cancelled"
    assert result.iterations == 0


def test_should_stop_mid_run_after_tool(tmp_path):
    stop = {"flag": False}

    def should_stop():
        return stop["flag"]

    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("write_file", {"path": "x.txt", "content": "hi"}),
            tool_call("git_status", {}),
            json.dumps({"done": True, "summary": "done"}),
        ],
    )
    loop.should_stop = should_stop

    # Operator presses STOP as soon as a modifying tool completes.
    def sink(event):
        if event.type == "tool_completed" and event.tool == "write_file":
            stop["flag"] = True

    loop.event_sink = sink
    result = loop.run("task")
    assert result.status == "cancelled"
    assert result.state == "cancelled"
    # The already-running write completed; the next step never ran.
    assert (tmp_path / "x.txt").exists()


def test_events_emitted_end_to_end(tmp_path):
    events = []
    sink = events.append

    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("git_status", {}),
            json.dumps({"done": True, "summary": "all good"}),
        ],
        config_overrides={"mode": "AUTO"},
    )
    loop.event_sink = sink
    result = loop.run("task")
    assert result.is_complete
    types = [e.type for e in events]
    assert "agent_started" in types
    assert "status" in types
    assert "agent_thinking" in types
    assert "tool_started" in types
    assert "tool_completed" in types
    assert "agent_completed" in types
    assert "mode_changed" in types


def test_test_command_emits_test_events(tmp_path):
    events = []
    sink = events.append
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("run_command", {"command": "python -m pytest"}),
            json.dumps({"done": True, "summary": "tests pass"}),
        ],
    )
    loop.event_sink = sink
    loop.run("task")
    types = [e.type for e in events]
    assert "command_started" in types
    assert "command_completed" in types
    assert "test_started" in types
    assert "test_completed" in types


def test_file_event_emitted_for_write(tmp_path):
    events = []
    sink = events.append
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("write_file", {"path": "a.txt", "content": "x"}),
            json.dumps({"done": True, "summary": "ok"}),
        ],
    )
    loop.event_sink = sink
    loop.run("task")
    types = [e.type for e in events]
    assert "file_written" in types


def test_plan_mode_build_states_via_verifying(tmp_path):
    from agent import state

    states = []
    loop, _ = make_loop(
        tmp_path,
        [
            tool_call("run_command", {"command": "pytest"}),
            json.dumps({"done": True, "summary": "ok"}),
        ],
    )
    loop.tracker.on_transition(lambda s, p: states.append(s))
    loop.run("task")
    assert state.VERIFYING in states


def test_is_test_command_variants():
    from agent.loop import is_test_command

    for cmd in (
        "pytest",
        "pytest tests/test_x.py",
        "python -m pytest",
        "py -m pytest",
        "pytest -q",
    ):
        assert is_test_command(cmd), cmd
    for cmd in ("python greet.py", "git status", "npm install"):
        assert not is_test_command(cmd), cmd
