"""Tests for the A.S.C.S. task-graph executor (agent.executor)."""

from __future__ import annotations

import json

import pytest

from agent.config import AgentConfig
from agent.executor import (
    TaskExecutor,
    TaskExecution,
    TaskOutcome,
    VerificationResult,
    task_system_prompt,
    task_user_prompt,
)
from agent.project import ProjectStore
from agent.tasks import COMPLETED, CANCELLED, FAILED, Task, build_graph_from_specs
from agent.workspace import Workspace


def _ok(executor, task):
    return TaskOutcome(task_id=task.id, ok=True, summary="done", iterations=1)


def _fail(executor, task):
    return TaskOutcome(
        task_id=task.id, ok=False, summary="boom", reason="a model boom"
    )


def _pass_verify(executor, task):
    return VerificationResult(task_id=task.id, ok=True, steps=[])


def _make(tmp_path, started=None, *, verify=None, store=True):
    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    client = None
    ws = Workspace(tmp_path)
    kwargs = dict(
        config=config,
        client=client,
        workspace=ws,
        event_sink=None,
        log=lambda m: None,
    )
    if store:
        kwargs["store"] = ProjectStore(tmp_path)
    if started is not None:
        kwargs["run_task"] = started
    if verify is not None:
        kwargs["verify"] = verify
    return TaskExecutor(**kwargs)


def _chain(titles):
    specs = []
    prev = None
    for i, t in enumerate(titles, start=1):
        specs.append({"id": f"T{i}", "title": t, "dependencies": [prev] if prev else []})
        prev = f"T{i}"
    return build_graph_from_specs(specs)


# -- prompts ---------------------------------------------------------------


def test_task_system_prompt_includes_task_and_verification():
    task = Task(id="T1", title="Add auth", verification=["run pytest"])
    prompt = task_system_prompt(task, "project", AgentConfig(workspace=".", mode="AUTO"))
    assert "Add auth" in prompt
    assert "T1" in prompt
    assert "run pytest" in prompt


def test_task_user_prompt():
    task = Task(id="T1", title="Add auth", description="Add login")
    msg = task_user_prompt(task)
    assert "Add auth" in msg
    assert "Add login" in msg


# -- orchestration ---------------------------------------------------------


def test_executes_sequential_tasks(tmp_path):
    executor = _make(tmp_path, started=_ok, verify=_pass_verify)
    graph = _chain(["A", "B", "C"])
    execution = executor.execute("job", graph)
    assert execution.is_success
    assert execution.status == "completed"
    assert len(execution.outcomes) == 3
    assert graph.all_complete
    assert all(g.status == COMPLETED for g in graph.tasks.values())


def test_exposes_failed_tasks_helper(tmp_path):
    executor = _make(tmp_path, started=_ok, verify=_pass_verify)
    execution = executor.execute("job", _chain(["A", "B"]))
    assert execution.completed_tasks == execution.outcomes
    assert execution.failed_tasks == []


def test_task_failure_marks_partial_and_cancels_dependents(tmp_path):
    calls = {}

    def started(executor, task):
        if task.id == "T2":
            return _fail(executor, task)
        return _ok(executor, task)

    executor = _make(tmp_path, started=started, verify=_pass_verify)
    graph = _chain(["A", "B", "C"])
    execution = executor.execute("job", graph)
    assert execution.status == "partial"
    assert graph.task("T2").status == FAILED
    assert graph.task("T3").status == CANCELLED  # blocked behind T2
    assert "T2" in execution.summary


def test_fan_in_fan_out_graph(tmp_path):
    executor = _make(tmp_path, started=_ok, verify=_pass_verify)
    graph = build_graph_from_specs(
        [
            {"id": "T1", "title": "Inspect"},
            {"id": "T2", "title": "Design", "dependencies": ["T1"]},
            {"id": "T3", "title": "Implement", "dependencies": ["T1"]},
            {"id": "T4", "title": "Verify", "dependencies": ["T2", "T3"]},
        ]
    )
    execution = executor.execute("job", graph)
    assert execution.is_success
    assert all(g.status == COMPLETED for g in graph.tasks.values())


def test_cancellation_stops(tmp_path):
    def stop():
        return True

    executor = _make(tmp_path, started=_ok, verify=_pass_verify)
    executor.should_stop = stop
    graph = _chain(["A", "B"])
    execution = executor.execute("job", graph)
    assert execution.status == "cancelled"


def test_persists_graph_state(tmp_path):
    store = ProjectStore(tmp_path)
    executor = _make(tmp_path, started=_ok, verify=_pass_verify, store=True)
    graph = _chain(["A", "B"])
    executor.execute("job", graph)
    loaded = store.load_task_graph()
    assert loaded is not None
    assert len(loaded) == 2


# -- default verification --------------------------------------------------


def test_verify_task_runs_commands_and_run_steps(tmp_path):
    executor = _make(tmp_path, started=_ok, store=False)
    task = Task(
        id="T1",
        title="X",
        commands=["python -c \"print('hi')\""],
        verification=["run python -c \"print('ok')\"", "confirm nothing changed"],
    )
    verification = executor.verify(executor, task)
    assert verification.ok
    statuses = [s["status"] for s in verification.steps]
    assert "ok" in statuses
    assert "noted" in statuses


def test_verify_task_fails_on_nonzero_exit(tmp_path):
    executor = _make(tmp_path, started=_ok, store=False)
    task = Task(
        id="T1",
        title="X",
        verification=["run python -c \"import sys; sys.exit(3)\""],
    )
    verification = executor.verify(executor, task)
    assert not verification.ok
    assert verification.steps[0]["status"] == "failed"


def test_verify_task_with_no_steps_passes(tmp_path):
    executor = _make(tmp_path, started=_ok, store=False)
    verification = executor.verify(executor, Task(id="T1", title="X"))
    assert verification.ok


# -- default model loop ----------------------------------------------------


class FakeClient:
    def __init__(self, responses, model="fake"):
        self.responses = list(responses)
        self.index = 0
        self.model = model

    def chat(self, messages, *, format="json", options=None, timeout=None):
        if self.index < len(self.responses):
            item = self.responses[self.index]
            self.index += 1
            return item
        raise AssertionError("FakeClient exhausted")


def _tool(name, args):
    return json.dumps({"tool": name, "arguments": args})


def test_default_run_task_reaches_done_then_verifies(tmp_path):
    client = FakeClient(
        [
            _tool("run_command", {"command": "python -c \"print('x')\""}),
            json.dumps({"done": True, "summary": "implemented"}),
        ]
    )
    executor = TaskExecutor(
        config=AgentConfig(workspace=tmp_path, mode="AUTO"),
        client=client,
        workspace=Workspace(tmp_path),
        log=lambda m: None,
    )
    outcome = executor.run_task(executor, Task(id="T1", title="X"))
    assert outcome.ok
    assert outcome.summary == "implemented"
    assert outcome.iterations == 2


def test_default_run_task_hits_iteration_limit(tmp_path):
    client = FakeClient(
        [_tool("git_status", {})] * 20
    )
    executor = TaskExecutor(
        config=AgentConfig(workspace=tmp_path, mode="AUTO", max_iterations=5),
        client=client,
        workspace=Workspace(tmp_path),
        log=lambda m: None,
    )
    outcome = executor.run_task(executor, Task(id="T1", title="X"))
    assert not outcome.ok
    assert "iterations" in outcome.reason


def test_task_execution_result_helpers():
    execution = TaskExecution(objective="x", status="completed")
    assert execution.is_success
    assert execution.failed_tasks == []
    assert execution.completed_tasks == []
