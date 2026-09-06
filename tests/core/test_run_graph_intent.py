"""run_graph intent-boundary regression tests (Phase 1 repair).

Covers the three repaired paths:

A. run_graph() no-work boundaries for conversation AND question intents —
   no task graph, no workspace refresh (.ascs), no mutation, no tools.
B. Malformed planner-graph recovery is intent-aware for every intent —
   a question must never become "Implement and verify".
C. The task executor enforces the same intent boundary as the single-shot
   loop (hostile tool calls inside tasks are refused for read-only intents).
"""

from __future__ import annotations

import json

import pytest

from agent.config import AgentConfig
from agent.core.intent import fallback_spec_for
from agent.execution.executor import TaskExecutor
from agent.execution.tasks import Task, TaskGraph
from agent.loop import AgentLoop
from agent.workspace import Workspace


def _done(summary):
    return json.dumps({"done": True, "summary": summary})


def _tool(tool, arguments, comment="step"):
    return json.dumps({"comment": comment, "tool": tool, "arguments": arguments})


class _ScriptedClient:
    """Replays scripted replies, then answers conversationally."""

    def __init__(self, script, model="fake-model"):
        self.script = list(script)
        self.index = 0
        self.model = model
        self.calls = []

    def chat(self, messages, *, format="json", options=None, timeout=None):
        self.calls.append(messages)
        if self.index < len(self.script):
            item = self.script[self.index]
            self.index += 1
            if isinstance(item, BaseException):
                raise item
            return item
        return _done("answered without tools")


class _HostileGraphClient:
    """Always tries to plan a mutating task, then always tries to write."""

    def __init__(self, model="fake-model"):
        self.model = model
        self.calls = 0

    def chat(self, messages, *, format="json", options=None, timeout=None):
        self.calls += 1
        prompt = messages[-1].get("content", "") if messages else ""
        if "Decompose the objective" in prompt or "planner" in prompt.lower():
            return json.dumps(
                {
                    "tasks": [
                        {
                            "id": "T1",
                            "title": "Write evil.txt",
                            "kind": "implement",
                            "verification": ["confirm done"],
                        }
                    ]
                }
            )
        return _tool("write_file", {"path": "evil.txt", "content": "evil"})


def _loop(tmp_path, client):
    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    return AgentLoop(config, client, Workspace(tmp_path), log=lambda m: None)


# ---------------------------------------------------------------------------
# A. run_graph no-work behavior for conversation + question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "thanks",
        "what is Python?",
        "what does authentication mean?",
    ],
)
def test_run_graph_no_work_requests_never_touch_workspace(tmp_path, text):
    loop = _loop(tmp_path, _ScriptedClient([_done("answer")]))
    result = loop.run_graph(text)
    assert result.is_complete
    assert result.task_count == 0  # no plan, no task graph
    assert list(tmp_path.iterdir()) == []  # no refresh, no .ascs, no files
    assert "(no tools were used)" in result.summary.lower()


def test_run_graph_question_skips_planner_entirely(tmp_path):
    """The planner model call must never happen for a high-confidence question."""
    client = _HostileGraphClient()
    loop = _loop(tmp_path, client)
    result = loop.run_graph("what is Python?")
    assert result.is_complete
    assert result.task_count == 0
    assert client.calls == 1  # only the conversational answer turn
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# B. Malformed planner-graph recovery stays intent-aware
# ---------------------------------------------------------------------------


def _cyclic_plan():
    return json.dumps(
        {
            "tasks": [
                {"id": "T1", "title": "First", "dependencies": ["T2"]},
                {"id": "T2", "title": "Second", "dependencies": ["T1"]},
            ]
        }
    )


@pytest.mark.parametrize(
    ("objective", "needle"),
    [
        ("what is Python?", "Review request"),
        ("show me the files in this project", "Inspect request"),
        ("verify the changes", "Verify request"),
    ],
)
def test_malformed_graph_fallback_never_implements_non_mutating(
    tmp_path, objective, needle
):
    """A cyclic (invalid) plan for a non-mutating intent must not implement."""
    loop = _loop(
        tmp_path,
        _ScriptedClient([_cyclic_plan(), _done("reported findings")]),
    )
    # Exercise the real _plan_objective recovery path directly.
    from agent.context.project import ProjectStore

    store = ProjectStore(tmp_path)
    planned = loop._plan_objective(objective, store)
    assert planned["count"] == 1
    assert needle in planned["text"]
    assert "Implement and verify" not in planned["text"]


def test_malformed_graph_fallback_for_work_stays_mutating_capable(tmp_path):
    loop = _loop(tmp_path, _ScriptedClient([_cyclic_plan()]))
    from agent.context.project import ProjectStore

    planned = loop._plan_objective("create example.py containing a calculator", ProjectStore(tmp_path))
    assert planned["count"] == 1
    assert "Implement and verify" in planned["text"]


def test_fallback_spec_kinds_cover_all_intents():
    assert fallback_spec_for("hello")["kind"] == "review"
    assert fallback_spec_for("what is Python?")["kind"] == "review"
    assert fallback_spec_for("show me the files in this project")["kind"] == "inspect"
    assert fallback_spec_for("delete foo.py")["kind"] == "implement"
    assert fallback_spec_for("run pytest")["kind"] == "verify"
    assert fallback_spec_for("verify the changes")["kind"] == "review"
    assert fallback_spec_for("Do the thing")["kind"] == "review"


# ---------------------------------------------------------------------------
# C. Executor intent gate mirrors the loop gate
# ---------------------------------------------------------------------------


def test_executor_blocks_mutation_for_inspection_intent(tmp_path):
    from agent.core.intent import classify_request

    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    ws = Workspace(tmp_path)
    executor = TaskExecutor(
        config=config,
        client=_ScriptedClient([]),
        workspace=ws,
        intent=classify_request("show me the files in this project"),
    )
    task = Task(id="T1", title="Inspect")
    blocked = executor._run_tool(
        "write_file", {"path": "evil.txt", "content": "evil"}, task, 1
    )
    assert not blocked.ok
    assert not (tmp_path / "evil.txt").exists()
    allowed = executor._run_tool("list_directory", {"path": "."}, task, 1)
    assert allowed.ok


def test_executor_without_intent_is_unrestricted_for_compat(tmp_path):
    """Direct construction without an intent keeps legacy behavior."""
    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    executor = TaskExecutor(
        config=config, client=_ScriptedClient([]), workspace=Workspace(tmp_path)
    )
    task = Task(id="T1", title="Work")
    result = executor._run_tool(
        "write_file", {"path": "ok.txt", "content": "ok"}, task, 1
    )
    assert result.ok
    assert (tmp_path / "ok.txt").exists()
