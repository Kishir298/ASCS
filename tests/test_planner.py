"""Tests for the A.S.C.S. planner (agent.planner)."""

from __future__ import annotations

import json

import pytest

from agent.planner import (
    parse_tasks,
    plan_objective,
    plan_text,
    planner_prompt,
    project_intelligence,
)
from agent.project import ProjectStore
from agent.tasks import COMPLETED


def _store(tmp_path) -> ProjectStore:
    return ProjectStore(tmp_path)


def _fake_chat(payload):
    def chat(messages, options):
        return json.dumps(payload)

    return chat


# -- parse_tasks -----------------------------------------------------------


def test_parse_tasks_handles_tasks_dict():
    specs = parse_tasks(
        {"tasks": [{"title": "A"}, {"title": "B"}]}
    )
    assert [s["title"] for s in specs] == ["A", "B"]


def test_parse_tasks_handles_bare_list():
    specs = parse_tasks(["T1", "T2"])
    assert [s["title"] for s in specs] == ["T1", "T2"]


def test_parse_tasks_handles_single_object():
    specs = parse_tasks({"title": "Only", "description": "desc"})
    assert len(specs) == 1
    assert specs[0]["title"] == "Only"


def test_parse_tasks_handles_string_with_lines():
    specs = parse_tasks("inspect\nimplement\ntest")
    assert [s["title"] for s in specs] == ["inspect", "implement", "test"]


def test_parse_tasks_filters_placeholders():
    specs = parse_tasks(
        {"tasks": [{"title": "Real"}, {"title": "No explicit plan"}]}
    )
    assert [s["title"] for s in specs] == ["Real"]


def test_parse_tasks_normalises_fields_and_complexity():
    specs = parse_tasks(
        [
            {
                "id": "T1",
                "title": "Auth",
                "description": "Add auth",
                "depends_on": ["T0"],
                "files": ["a.py", "b.py"],
                "command": ["pytest"],
                "verify": ["run pytest"],
                "size": "big",
                "kind": "implement",
            }
        ]
    )
    spec = specs[0]
    assert spec["id"] == "T1"
    assert spec["files"] == ["a.py", "b.py"]
    assert spec["commands"] == ["pytest"]
    assert spec["verification"] == ["run pytest"]
    assert spec["complexity"] == "large"
    assert spec["kind"] == "implement"
    assert spec["dependencies"] == ["T0"]


def test_parse_tasks_skips_non_dict_entries():
    specs = parse_tasks({"tasks": [42, None, "ok"]})
    assert [s["title"] for s in specs] == ["ok"]


def test_parse_tasks_returns_empty_for_none():
    assert parse_tasks(None) == []


# -- planner_prompt / project_intelligence --------------------------------


def test_planner_prompt_mentions_objective_and_rules():
    prompt = planner_prompt("Add auth", "project info")
    assert "Add auth" in prompt
    assert "dependencies" in prompt
    assert "verification" in prompt


def test_project_intelligence_mentions_project_name(tmp_path):
    store = _store(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n", encoding="utf-8"
    )
    text = project_intelligence(store, "add auth")
    assert "Project" in text


# -- plan_objective --------------------------------------------------------


def test_plan_objective_builds_dag_from_model(tmp_path):
    store = _store(tmp_path)
    payload = {
        "tasks": [
            {"id": "T1", "title": "Inspect API", "kind": "inspect"},
            {"id": "T2", "title": "Design config", "dependencies": ["T1"], "kind": "plan"},
            {"id": "T3", "title": "Implement service", "dependencies": ["T1"], "kind": "implement"},
            {"id": "T4", "title": "Verify", "dependencies": ["T2", "T3"], "kind": "verify"},
        ]
    }
    graph = plan_objective(store, "Add auth", _fake_chat(payload))
    assert len(graph) == 4
    assert graph.task("T4").dependencies == ["T2", "T3"]
    assert [t.id for t in graph.ordered()] == ["T1", "T2", "T3", "T4"]
    # Only the root is ready.
    assert graph.ready_tasks() and graph.ready_tasks()[0].id == "T1"


def test_plan_objective_auto_splits_large_task(tmp_path):
    store = _store(tmp_path)
    payload = {
        "tasks": [
            {
                "id": "BIG",
                "title": "Implement everything",
                "complexity": "large",
                "files": ["a.py", "b.py", "c.py"],
                "commands": ["pytest"],
                "verification": ["run pytest"],
            }
        ]
    }
    graph = plan_objective(store, "big job", _fake_chat(payload))
    # One large task with 3 files -> split into subtasks.
    assert len(graph) > 1
    assert all(t.complexity != "large" for t in graph.tasks.values())


def test_plan_objective_falls_back_when_no_tasks(tmp_path):
    store = _store(tmp_path)
    graph = plan_objective(store, "Do the thing", _fake_chat({"unrelated": True}))
    assert len(graph) == 1
    task = next(iter(graph.tasks.values()))
    assert "Do the thing" in task.title
    assert task.verification  # fallback guarantees verification


def test_plan_objective_guarantees_verification_everywhere(tmp_path):
    store = _store(tmp_path)
    payload = {
        "tasks": [
            {"id": "A", "title": "Implement", "files": ["x.py"], "kind": "implement"}
        ]
    }
    graph = plan_objective(store, "job", _fake_chat(payload))
    assert all(t.verification for t in graph.tasks.values())


def test_plan_objective_respects_provided_verification(tmp_path):
    store = _store(tmp_path)
    payload = {
        "tasks": [
            {
                "id": "A",
                "title": "Implement",
                "verification": ["run pytest tests/auth/"],
                "kind": "implement",
            }
        ]
    }
    graph = plan_objective(store, "job", _fake_chat(payload))
    assert graph.task("A").verification == ["run pytest tests/auth/"]


def test_plan_objective_readiness_chains(tmp_path):
    store = _store(tmp_path)
    payload = {
        "tasks": [
            {"id": "T1", "title": "One"},
            {"id": "T2", "title": "Two", "dependencies": ["T1"]},
            {"id": "T3", "title": "Three", "dependencies": ["T2"]},
        ]
    }
    graph = plan_objective(store, "job", _fake_chat(payload))
    graph.mark("T1", COMPLETED)
    assert {t.id for t in graph.ready_tasks()} == {"T2"}
    graph.mark("T2", COMPLETED)
    assert {t.id for t in graph.ready_tasks()} == {"T3"}


# -- plan_text -------------------------------------------------------------


def test_plan_text_renders_ordered_plan(tmp_path):
    store = _store(tmp_path)
    payload = {
        "tasks": [
            {"id": "T1", "title": "Inspect"},
            {"id": "T2", "title": "Implement", "dependencies": ["T1"], "verification": ["run pytest"]},
        ]
    }
    graph = plan_objective(store, "job", _fake_chat(payload))
    text = plan_text(graph)
    assert "Inspect" in text
    assert "Implement" in text
    assert "run pytest" in text


def test_parse_tasks_dedupes_dependencies():
    specs = parse_tasks(
        [{"title": "A", "dependencies": ["T1", "T1", "T2"]}]
    )
    assert specs[0]["dependencies"] == ["T1", "T2"]