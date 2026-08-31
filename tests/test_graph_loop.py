"""Tests for the task-graph-driven loop path (AgentLoop.run_graph)."""

from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.loop import AgentLoop, GraphLoopResult
from agent.project import ProjectStore
from agent.tasks import COMPLETED, Task, TaskGraph
from agent.workspace import Workspace


def _collect():
    events = []

    def sink(event):
        events.append(event)

    return events, sink


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
        raise AssertionError("FakeClient exhausted scripted responses")


def _make(tmp_path, responses):
    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    client = FakeClient(responses, model=config.model)
    ws = Workspace(tmp_path)
    return config, client, ws


def _plan_single_ok_task():
    return json.dumps(
        {
            "tasks": [
                {
                    "id": "T1",
                    "title": "Write hello",
                    "kind": "implement",
                    "verification": ["run python -c \"print('hi')\""],
                }
            ]
        }
    )


def test_run_graph_plans_executes_and_persists(tmp_path):
    config, client, ws = _make(
        tmp_path,
        [
            _plan_single_ok_task(),
            json.dumps({"done": True, "summary": "implemented hello"}),
        ],
    )
    events, sink = _collect()
    loop = AgentLoop(config, client, ws, log=lambda m: None, event_sink=sink)
    result = loop.run_graph("Write hello")
    assert result.status == "completed"
    assert result.is_complete
    assert "Write hello" in result.plan
    assert result.task_count == 1
    # Persisted graph is complete.
    store = ProjectStore(tmp_path)
    loaded = store.load_task_graph()
    assert loaded is not None
    assert loaded.task("T1").status == COMPLETED
    # Task-plan inspection event emitted.
    assert any(e.type == "task_plan" for e in events)


def test_run_graph_falls_back_when_planner_returns_nothing(tmp_path):
    config, client, ws = _make(
        tmp_path,
        [
            json.dumps({"tasks": []}),
            json.dumps({"done": True, "summary": "wrote it"}),
        ],
    )
    loop = AgentLoop(config, client, ws, log=lambda m: None)
    result = loop.run_graph("Do the thing")
    assert result.status == "completed"
    assert result.task_count == 1


def test_run_graph_resumes_persisted_graph(tmp_path):
    # Pre-seed a graph where T1 is already completed and T2 is pending.
    store = ProjectStore(tmp_path)
    graph = TaskGraph()
    graph.add(Task(id="T1", title="Done", status=COMPLETED, verification=["run python -c \"print(1)\""]))
    graph.add(Task(id="T2", title="Also do", dependencies=["T1"], verification=["run python -c \"print(2)\""]))
    store.save_task_graph(graph)

    config, client, ws = _make(
        tmp_path,
        [
            _plan_single_ok_task(),  # planner still runs even when resuming
            json.dumps({"done": True, "summary": "finished T2"}),
        ],
    )
    loop = AgentLoop(config, client, ws, log=lambda m: None)
    result = loop.run_graph("Continue", resume=True)
    assert result.status == "completed"
    loaded = store.load_task_graph()
    assert loaded.task("T2").status == COMPLETED


def test_run_graph_cancels_when_stopped(tmp_path):
    config, client, ws = _make(
        tmp_path,
        [
            _plan_single_ok_task(),
        ],
    )
    loop = AgentLoop(config, client, ws, log=lambda m: None, should_stop=lambda: True)
    result = loop.run_graph("Write hello")
    assert result.status == "cancelled"


def test_run_graph_returns_graph_result_shape(tmp_path):
    config, client, ws = _make(tmp_path, [])
    result = GraphLoopResult(status="completed", objective="x")
    assert isinstance(result, GraphLoopResult)
    assert result.is_complete
