"""Tests for the A.S.C.S. task engine (agent.tasks)."""

from __future__ import annotations

import pytest

from agent.models import Plan
from agent.tasks import (
    BLOCKED,
    CANCELLED,
    COMPLETED,
    FAILED,
    PENDING,
    READY,
    RUNNING,
    SKIPPED,
    VALID_STATUSES,
    Task,
    TaskGraph,
    TaskGraphError,
    plan_to_graph,
)


def test_task_defaults_to_pending():
    task = Task(id="t1", title="first")
    assert task.status == PENDING
    assert task.is_terminal is False


def test_terminal_statuses_present():
    assert COMPLETED in VALID_STATUSES
    assert {COMPLETED, FAILED, CANCELLED, SKIPPED} <= VALID_STATUSES


def test_graph_adds_and_retrieves():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    assert graph.task("a").title == "A"
    with pytest.raises(TaskGraphError):
        graph.add(Task(id="a", title="dup"))


def test_graph_unknown_task():
    graph = TaskGraph()
    with pytest.raises(TaskGraphError):
        graph.task("nope")


def test_duplicate_id_raises():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    with pytest.raises(TaskGraphError):
        graph.add(Task(id="a", title="A2"))


def test_unknown_dependency_raises_on_validate():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A", dependencies=["ghost"]))
    with pytest.raises(TaskGraphError):
        graph.validate()


def test_cycle_detected():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A", dependencies=["b"]))
    graph.add(Task(id="b", title="B", dependencies=["a"]))
    with pytest.raises(TaskGraphError):
        graph.validate()


def test_ordered_puts_dependencies_first():
    graph = TaskGraph()
    graph.add(Task(id="b", title="B", dependencies=["a"]))
    graph.add(Task(id="a", title="A"))
    ordered = graph.ordered()
    assert [t.id for t in ordered] == ["a", "b"]


def test_ready_tasks_only_when_dependencies_ready():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    graph.add(Task(id="b", title="B", dependencies=["a"]))
    assert [t.id for t in graph.ready_tasks()] == ["a"]
    graph.mark("a", COMPLETED)
    assert [t.id for t in graph.ready_tasks()] == ["b"]


def test_mark_cascades_blocked_on_failed_dependency():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    graph.add(Task(id="b", title="B", dependencies=["a"]))
    graph.add(Task(id="c", title="C", dependencies=["b"]))
    graph.mark("a", FAILED)
    graph.recompute_statuses()
    assert graph.task("b").status == BLOCKED
    assert graph.task("c").status == BLOCKED


def test_invalid_status_rejected():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    with pytest.raises(TaskGraphError):
        graph.mark("a", "done_ish")


def test_progress_counts():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    graph.add(Task(id="b", title="B"))
    graph.mark("a", COMPLETED)
    progress = graph.progress()
    assert progress["total"] == 2
    assert progress["completed"] == 1


def test_all_complete():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    graph.add(Task(id="b", title="B"))
    graph.mark("a", COMPLETED)
    assert not graph.all_complete
    graph.mark("b", COMPLETED)
    assert graph.all_complete


def test_roundtrip_via_dict(tmp_path):
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    graph.add(Task(id="b", title="B", dependencies=["a"]))
    payload = graph.to_dict()
    restored = TaskGraph.from_dict(payload)
    assert len(restored) == 2
    assert restored.task("b").dependencies == ["a"]


def test_plan_to_graph_sequential():
    plan = Plan(["inspect", "implement", "test"], goal="goal")
    graph = plan_to_graph(plan)
    assert len(graph) == 3
    ids = [t.id for t in graph.ordered()]
    assert ids == ["task-1", "task-2", "task-3"]
    # sequential dependency chain
    assert graph.task("task-2").dependencies == ["task-1"]
    assert graph.task("task-3").dependencies == ["task-2"]
    # only the first task is ready
    assert [t.id for t in graph.ready_tasks()] == ["task-1"]


def test_plan_to_graph_ignores_placeholder_steps():
    plan = Plan(["No explicit plan provided."])
    graph = plan_to_graph(plan)
    assert len(graph) == 1
    assert graph.task("task-1").status == READY