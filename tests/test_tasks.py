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
    build_graph_from_specs,
    chunk_graph,
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


# -- T2: structured DAG support -----------------------------------------


def test_task_has_complexity_and_kind_defaults():
    task = Task(id="t1", title="first")
    assert task.complexity == "medium"
    assert task.kind == "implement"
    assert task.is_small is False


def test_task_small_complexity_property():
    assert Task(id="a", title="a", complexity="small").is_small is True


def test_build_graph_from_specs_preserves_fan_out():
    specs = [
        {"id": "T1", "title": "Inspect"},
        {"id": "T2", "title": "Design", "dependencies": ["T1"]},
        {"id": "T3", "title": "Implement", "dependencies": ["T1"]},
        {"id": "T4", "title": "Verify", "dependencies": ["T2", "T3"]},
    ]
    graph = build_graph_from_specs(specs)
    assert len(graph) == 4
    assert graph.task("T4").dependencies == ["T2", "T3"]
    # T2 and T3 both depend on T1 (fan-out) and are ready together after T1.
    graph.mark("T1", COMPLETED)
    assert {t.id for t in graph.ready_tasks()} == {"T2", "T3"}
    assert {t.id for t in graph.ordered()} == {"T1", "T2", "T3", "T4"}


def test_build_graph_from_specs_auto_ids():
    specs = [{"title": "A"}, {"title": "B"}]
    graph = build_graph_from_specs(specs, prefix="work")
    assert graph.task("work-1").title == "A"
    assert graph.task("work-2").title == "B"
    assert len(graph) == 2


def test_build_graph_from_specs_accepts_task_objects():
    graph = build_graph_from_specs(
        [Task(id="a", title="A"), Task(id="b", title="B", dependencies=["a"])]
    )
    assert len(graph) == 2
    assert graph.task("b").dependencies == ["a"]


def test_build_graph_from_specs_rejects_bad_spec():
    with pytest.raises(TaskGraphError):
        build_graph_from_specs([42])


def test_build_graph_from_specs_detects_cycle():
    specs = [
        {"id": "a", "title": "A", "dependencies": ["b"]},
        {"id": "b", "title": "B", "dependencies": ["a"]},
    ]
    with pytest.raises(TaskGraphError):
        build_graph_from_specs(specs)


def test_build_graph_from_specs_detects_unknown_dep():
    with pytest.raises(TaskGraphError):
        build_graph_from_specs(
            [{"id": "a", "title": "A", "dependencies": ["ghost"]}]
        )


def test_tasks_by_status_and_failed():
    graph = TaskGraph()
    graph.add(Task(id="a", title="A"))
    graph.add(Task(id="b", title="B"))
    graph.mark("a", FAILED)
    assert graph.tasks_by_status(FAILED) == [graph.task("a")]
    assert graph.failed == [graph.task("a")]
    assert graph.task("b").status in (PENDING, READY)


def test_first_failure_is_precedence_order():
    graph = TaskGraph()
    graph.add(Task(id="T1", title="1"))
    graph.add(Task(id="T2", title="2", dependencies=["T1"]))
    graph.add(Task(id="T3", title="3", dependencies=["T2"]))
    graph.mark("T2", FAILED)
    assert graph.first_failure.id == "T2"


def test_cascade_cancel_blocks_dependents():
    graph = TaskGraph()
    graph.add(Task(id="T1", title="1"))
    graph.add(Task(id="T2", title="2", dependencies=["T1"]))
    graph.add(Task(id="T3", title="3", dependencies=["T2"]))
    graph.add(Task(id="IND", title="independent"))
    graph.mark("T1", FAILED)
    cancelled = graph.cascade_cancel()
    assert graph.task("T2").status == CANCELLED
    assert graph.task("T3").status == CANCELLED
    assert "IND" not in cancelled
    assert graph.task("IND").status in (PENDING, READY)


def test_chunk_graph_splits_large_task_by_files():
    spec = Task(
        id="big",
        title="Implement everything",
        complexity="large",
        files=["a.py", "b.py", "c.py"],
        commands=["pytest"],
        verification=["pytest"],
    )
    graph = build_graph_from_specs([spec])
    split = chunk_graph(graph)
    assert len(split) == 4  # 3 files + 1 commands
    ids = sorted(split.tasks)
    assert ids == ["big.1", "big.2", "big.3", "big.4"]
    # Deterministic sequential dependency among subtasks.
    assert split.task("big.2").dependencies == ["big.1"]
    assert split.task("big.3").dependencies == ["big.2"]
    # Each subtask is medium complexity (no longer "large").
    assert all(split.tasks[i].complexity == "medium" for i in ids)


def test_chunk_graph_leaves_medium_tasks_alone():
    graph = build_graph_from_specs(
        [{"id": "a", "title": "A", "complexity": "medium"}]
    )
    split = chunk_graph(graph)
    assert len(split) == 1
    assert split.task("a").id == "a"


def test_chunk_graph_splits_standalone_large_with_no_files():
    graph = build_graph_from_specs(
        [{"id": "x", "title": "X", "complexity": "large", "files": []}]
    )
    split = chunk_graph(graph)
    assert len(split) == 1
    assert split.task("x.1").id == "x.1"