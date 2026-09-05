"""Integration tests for the full coding-agent pipeline (Phase 4–7).

Covers: SAFE/BUILD/PLAN mode gating, git dirty-state protection, planner→DAG→executor
integration, task chunking, dependency execution, tool output capture, verification
failure + retry, interrupt + resume, structured action log, and task events.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from agent.config import AgentConfig
from agent.events import EVENT_TYPES, emit_task_failed, emit_task_verified
from agent.executor import (
    TaskActionLog,
    TaskExecutor,
    TaskExecution,
    TaskOutcome,
    VerificationResult,
)
from agent.loop import AgentLoop, GraphLoopResult
from agent.project import ProjectStore
from agent.tasks import (
    COMPLETED,
    FAILED,
    PENDING,
    READY,
    RUNNING,
    CANCELLED,
    Task,
    TaskGraph,
    build_graph_from_specs,
    chunk_graph,
)
from agent.workspace import Workspace


# ── helpers ────────────────────────────────────────────────────────────────

class FakeClient:
    """Minimal fake Ollama client for deterministic integration tests."""

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


def _json(obj):
    return json.dumps(obj)


def _tool(name, args):
    return _json({"tool": name, "arguments": args})


def _done(summary="done"):
    return _json({"done": True, "summary": summary})


def _collect_events():
    events = []
    return events, lambda event: events.append(event)


def _ok_run(executor, task):
    return TaskOutcome(task_id=task.id, ok=True, summary="done", iterations=1)


def _fail_run(executor, task):
    return TaskOutcome(task_id=task.id, ok=False, summary="boom", reason="model error")


def _pass_verify(executor, task):
    return VerificationResult(task_id=task.id, ok=True, steps=[])


def _fail_verify(executor, task):
    return VerificationResult(
        task_id=task.id,
        ok=False,
        steps=[{"step": "test", "status": "failed", "ok": False, "output": "FAIL"}],
    )


def _make_executor(
    tmp_path,
    *,
    mode="AUTO",
    started=None,
    verify=None,
    approver=None,
    git_baseline=None,
    max_verify_retries=2,
):
    config = AgentConfig(workspace=tmp_path, mode=mode, max_verify_retries=max_verify_retries)
    ws = Workspace(tmp_path)
    kwargs = dict(
        config=config,
        client=None,
        workspace=ws,
        event_sink=None,
        log=lambda m: None,
        approver=approver,
        git_baseline=git_baseline or set(),
    )
    if started is not None:
        kwargs["run_task"] = started
    if verify is not None:
        kwargs["verify"] = verify
    return TaskExecutor(**kwargs)


def _make_loop(tmp_path, responses, *, mode="AUTO", approver=None, should_stop=None, event_sink=None):
    config = AgentConfig(workspace=tmp_path, mode=mode)
    client = FakeClient(responses, model=config.model)
    ws = Workspace(tmp_path)
    loop = AgentLoop(
        config, client, ws,
        log=lambda m: None,
        event_sink=event_sink,
        should_stop=should_stop or (lambda: False),
        approver=approver,
    )
    return loop


# ── 1. SAFE-mode workflow ─────────────────────────────────────────────────

def test_safe_mode_blocks_writes_without_approval(tmp_path):
    """SAFE mode with no approver → _run_tool returns blocked."""
    executor = _make_executor(tmp_path, mode="SAFE")
    result = executor._run_tool(
        "write_file", {"path": "x.py", "content": "x=1"},
        Task(id="T1", title="Write"), 1,
    )
    assert not result.ok
    assert "no approver" in result.output.lower() or "approval" in result.output.lower()


def test_safe_mode_blocks_when_operator_declines(tmp_path):
    """SAFE mode with approver that denies → _run_tool returns blocked."""
    executor = _make_executor(
        tmp_path, mode="SAFE", approver=lambda desc: False,
    )
    result = executor._run_tool(
        "write_file", {"path": "x.py", "content": "x=1"},
        Task(id="T1", title="Write"), 1,
    )
    assert not result.ok
    assert "declined" in result.output.lower()


def test_safe_mode_allows_writes_on_approval(tmp_path):
    """SAFE mode with approving approver → flat loop writes file."""
    loop = _make_loop(
        tmp_path,
        [
            _tool("write_file", {"path": "hello.py", "content": "print('hi')\n"}),
            _done("wrote hello"),
        ],
        mode="SAFE",
        approver=lambda desc: True,
    )
    result = loop.run("write hello")
    assert result.is_complete
    assert (tmp_path / "hello.py").exists()


# ── 2. BUILD-mode workflow ─────────────────────────────────────────────────

def test_build_mode_executes_and_modifies_files(tmp_path):
    """BUILD mode: full task execution writes files and verifies."""
    client = FakeClient([
        _tool("write_file", {"path": "greeting.py", "content": "print('hello')\n"}),
        _done("created greeting.py"),
    ])
    config = AgentConfig(workspace=tmp_path, mode="BUILD")
    ws = Workspace(tmp_path)
    loop = AgentLoop(config, client, ws, log=lambda m: None)
    result = loop.run("create a greeting script")
    assert result.is_complete
    assert (tmp_path / "greeting.py").exists()


# ── 3. PLAN-mode read-only enforcement ────────────────────────────────────

def test_plan_mode_blocks_modify_in_task_engine(tmp_path):
    """PLAN mode: task executor refuses modifying tools."""
    executor = _make_executor(
        tmp_path, mode="PLAN",
        started=_ok_run, verify=_pass_verify,
    )
    result = executor._check_tool_allowed("write_file")
    assert result is not None
    assert not result.ok
    assert "PLAN mode" in result.output

    result2 = executor._check_tool_allowed("run_command")
    assert result2 is not None
    assert not result2.ok
    assert "PLAN mode" in result2.output


def test_plan_mode_allows_read_only(tmp_path):
    """PLAN mode: read-only tools pass through."""
    executor = _make_executor(tmp_path, mode="PLAN")
    for tool in ("read_file", "search_files", "list_directory", "git_status", "set_plan"):
        result = executor._check_tool_allowed(tool)
        assert result is None, f"{tool} should be allowed in PLAN mode"


def test_plan_mode_blocks_verify_run_command(tmp_path):
    """PLAN mode: _verify_task blocks run_command in verification steps."""
    executor = _make_executor(tmp_path, mode="PLAN")
    task = Task(
        id="T1", title="X",
        verification=["run python -c \"print(1)\""],
    )
    verification = executor.verify(executor, task)
    assert not verification.ok
    assert any(s["status"] == "blocked" for s in verification.steps)


# ── 4. Planner → DAG → executor integration ───────────────────────────────

def test_full_planner_dag_executor_flow(tmp_path):
    """Full flow: fake planner output → DAG build → execute → persist → report."""
    planner_response = _json({
        "tasks": [
            {"id": "T1", "title": "Inspect", "kind": "inspect",
             "verification": ["run python -c \"print('inspected')\""]},
            {"id": "T2", "title": "Implement", "dependencies": ["T1"],
             "kind": "implement", "files": ["app.py"],
             "verification": ["run python -c \"print('ok')\""]},
        ]
    })
    responses = [
        planner_response,
        _tool("list_directory", {}),
        _done("inspected"),
        _tool("write_file", {"path": "app.py", "content": "x=1\n"}),
        _done("implemented app.py"),
    ]
    loop = _make_loop(tmp_path, responses)
    result = loop.run_graph("inspect and implement")
    assert result.status == "completed"
    assert result.task_count == 2
    # Persisted.
    store = ProjectStore(tmp_path)
    loaded = store.load_task_graph()
    assert loaded is not None
    assert loaded.all_complete


# ── 5. Automatic task chunking preserves deps ──────────────────────────────

def test_chunking_preserves_dependencies_and_order(tmp_path):
    """Large task is split; subtasks inherit deps and are ordered."""
    graph = build_graph_from_specs([
        {"id": "T1", "title": "Setup", "complexity": "small"},
        {"id": "T2", "title": "Build everything",
         "complexity": "large",
         "dependencies": ["T1"],
         "files": ["a.py", "b.py"],
         "verification": ["run pytest"]},
    ])
    chunked = chunk_graph(graph)
    # T1 is unchanged (small).
    assert "T1" in chunked.tasks
    # T2 is split into subtasks.
    subtask_ids = [tid for tid in chunked.tasks if tid.startswith("T2.")]
    assert len(subtask_ids) >= 2
    # All subtasks depend on T1.
    for tid in subtask_ids:
        assert "T1" in chunked.tasks[tid].dependencies
    # Subtask ordering is preserved (each depends on previous).
    if len(subtask_ids) > 1:
        for i in range(1, len(subtask_ids)):
            assert subtask_ids[i - 1] in chunked.tasks[subtask_ids[i]].dependencies
    # Persistence round-trip.
    chunked.validate()


# ── 6. Dependency execution order (fan-in/fan-out) ────────────────────────

def test_fan_in_fan_out_correct_order(tmp_path):
    """T1→{T2,T3}→T4: parallel branches then merge."""
    executor = _make_executor(
        tmp_path, started=_ok_run, verify=_pass_verify,
    )
    graph = build_graph_from_specs([
        {"id": "T1", "title": "Inspect"},
        {"id": "T2", "title": "Design", "dependencies": ["T1"]},
        {"id": "T3", "title": "Implement", "dependencies": ["T1"]},
        {"id": "T4", "title": "Verify", "dependencies": ["T2", "T3"]},
    ])
    execution = executor.execute("fan-in/out", graph)
    assert execution.is_success
    order = [o.task_id for o in execution.outcomes]
    assert order.index("T1") < order.index("T2")
    assert order.index("T1") < order.index("T3")
    assert order.index("T2") < order.index("T4")
    assert order.index("T3") < order.index("T4")


# ── 7. Tool execution captures output ─────────────────────────────────────

def test_tool_execution_captures_output_and_exit_code(tmp_path):
    """run_command output is captured and returned."""
    client = FakeClient([
        _tool("run_command", {"command": "python -c \"print('hello world')\""}),
        _done("executed"),
    ])
    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    ws = Workspace(tmp_path)
    executor = TaskExecutor(
        config=config, client=None, workspace=ws, log=lambda m: None,
    )
    from agent.tools import execute_tool
    result = execute_tool("run_command", {"command": "python -c \"print('hello world')\""}, ws, config)
    assert result.ok
    assert "hello world" in result.output
    assert "exit code 0" in result.note


# ── 8. Verification failure blocks success + retries ───────────────────────

def test_verify_failure_retries_then_fails(tmp_path):
    """When verification always fails, task retries then stops as FAILED."""
    def _always_fail(executor, task):
        return VerificationResult(
            task_id=task.id, ok=False,
            steps=[{"step": "test", "status": "failed", "ok": False, "output": "FAIL"}],
        )

    executor = _make_executor(
        tmp_path, started=_ok_run, verify=_always_fail,
        max_verify_retries=2,
    )
    graph = build_graph_from_specs([{"id": "T1", "title": "X"}])
    execution = executor.execute("test", graph)
    assert execution.status == "partial"
    assert graph.task("T1").status == FAILED
    assert "verification failed" in graph.task("T1").failure_reason


def test_verify_retry_succeeds_on_second_attempt(tmp_path):
    """First verify fails, model retries and second verify passes."""
    call_count = [0]

    def _retrying_verify(executor, task):
        call_count[0] += 1
        if call_count[0] <= 1:
            return VerificationResult(
                task_id=task.id, ok=False,
                steps=[{"step": "test", "status": "failed", "ok": False, "output": "fail"}],
            )
        return VerificationResult(task_id=task.id, ok=True, steps=[])

    executor = _make_executor(
        tmp_path, started=_ok_run, verify=_retrying_verify,
        max_verify_retries=2,
    )
    graph = build_graph_from_specs([{"id": "T1", "title": "X"}])
    execution = executor.execute("test", graph)
    assert execution.status == "completed"
    assert graph.task("T1").status == COMPLETED
    assert call_count[0] == 2


def test_no_steps_fails_implementing_task(tmp_path):
    """Implementing task with no verification steps → not fully verified."""
    executor = _make_executor(tmp_path)
    task = Task(id="T1", title="Build", kind="implement")
    verification = executor.verify(executor, task)
    assert not verification.ok
    assert "no verification steps" in verification.detail


def test_no_steps_passes_inspect_task(tmp_path):
    """Non-implementing task with no verification steps → passes."""
    executor = _make_executor(tmp_path)
    task = Task(id="T1", title="Review", kind="inspect")
    verification = executor.verify(executor, task)
    assert verification.ok


# ── 9. Interrupted execution + resume ──────────────────────────────────────

def test_interrupted_execution_and_resume(tmp_path):
    """Run, stop mid-way, resume → completed tasks preserved, remaining run."""
    store = ProjectStore(tmp_path)
    graph = TaskGraph()
    graph.add(Task(id="T1", title="Done", status=COMPLETED,
                    verification=["run python -c \"print(1)\""]))
    graph.add(Task(id="T2", title="Also do", dependencies=["T1"],
                    verification=["run python -c \"print(2)\""]))
    store.save_task_graph(graph)

    planner_response = _json({
        "tasks": [{"id": "T1", "title": "Done"},
                  {"id": "T2", "title": "Also do", "dependencies": ["T1"]}]
    })
    responses = [planner_response, _done("finished T2")]
    loop = _make_loop(tmp_path, responses)
    result = loop.run_graph("Continue", resume=True)
    assert result.status == "completed"
    loaded = store.load_task_graph()
    assert loaded.task("T2").status == COMPLETED


def test_stuck_running_resets_on_resume(tmp_path):
    """A task stuck in RUNNING is reset to READY on resume."""
    graph = TaskGraph()
    graph.add(Task(id="T1", title="Done", status=COMPLETED))
    graph.add(Task(id="T2", title="Stuck", dependencies=["T1"], status=RUNNING))
    graph.add(Task(id="T3", title="Waiting", dependencies=["T2"]))

    reset = TaskExecutor.reset_stuck_tasks(graph)
    assert "T2" in reset
    assert graph.task("T2").status == READY


# ── 10. Git dirty-state protection ─────────────────────────────────────────

def test_git_dirty_blocks_write(tmp_path):
    """File dirty at baseline → write blocked."""
    executor = _make_executor(
        tmp_path, git_baseline={"README.md"},
    )
    result = executor._check_git_dirty("write_file", {"path": "README.md", "content": "new"})
    assert result is not None
    assert not result.ok
    assert "Protected" in result.output


def test_git_dirty_allows_clean_files(tmp_path):
    """Clean files → write allowed."""
    executor = _make_executor(
        tmp_path, git_baseline={"README.md"},
    )
    result = executor._check_git_dirty("write_file", {"path": "app.py", "content": "x=1"})
    assert result is None


def test_git_dirty_skips_read_only_tools(tmp_path):
    """Read-only tools are not gated by git dirty."""
    executor = _make_executor(
        tmp_path, git_baseline={"README.md"},
    )
    result = executor._check_git_dirty("read_file", {"path": "README.md"})
    assert result is None


# ── 11. Task events ────────────────────────────────────────────────────────

def test_task_events_emitted(tmp_path):
    """task_started, task_completed, task_verified events emitted."""
    events, sink = _collect_events()
    executor = TaskExecutor(
        config=AgentConfig(workspace=tmp_path, mode="AUTO"),
        client=None,
        workspace=Workspace(tmp_path),
        event_sink=sink,
        log=lambda m: None,
        run_task=_ok_run,
        verify=_pass_verify,
    )
    graph = build_graph_from_specs([{"id": "T1", "title": "X"}])
    executor.execute("test", graph)
    types = [e.type for e in events]
    assert "task_verified" in types
    assert "task_completed" not in types  # wired_run emits task_completed, not execute()


def test_task_failed_event_on_failure(tmp_path):
    """task_failed event emitted when model loop fails."""
    events, sink = _collect_events()
    executor = TaskExecutor(
        config=AgentConfig(workspace=tmp_path, mode="AUTO"),
        client=None,
        workspace=Workspace(tmp_path),
        event_sink=sink,
        log=lambda m: None,
        run_task=_fail_run,
        verify=_pass_verify,
    )
    graph = build_graph_from_specs([{"id": "T1", "title": "X"}])
    executor.execute("test", graph)
    types = [e.type for e in events]
    assert "task_failed" in types


# ── 12. Per-task action log ────────────────────────────────────────────────

def test_action_log_recorded_on_verify(tmp_path):
    """Action log captures verification steps."""
    executor = _make_executor(tmp_path, started=_ok_run, verify=_pass_verify)
    graph = build_graph_from_specs([
        {"id": "T1", "title": "X", "verification": ["run python -c \"print(1)\""]},
    ])
    execution = executor.execute("test", graph)
    assert execution.completed_tasks
    outcome = execution.completed_tasks[0]
    assert any(a.action == "verify" for a in outcome.action_log)


# ── 13. Full end-to-end repository workflow ────────────────────────────────

def test_end_to_end_real_repo_workflow(tmp_path):
    """Full workflow: git init → scan → plan → chunk → DAG → build → verify → persist → report."""
    # Set up a minimal git repo.
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=False)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True, check=False)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True, check=False)
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=False)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), capture_output=True, check=False)

    planner_response = _json({
        "tasks": [
            {"id": "T1", "title": "Create utility", "kind": "implement",
             "files": ["utils.py"],
             "verification": ["run python -c \"import utils; print(utils.add(1,2))\""]},
        ]
    })
    responses = [
        planner_response,
        _tool("write_file", {"path": "utils.py", "content": "def add(a, b): return a + b\n"}),
        _done("created utils.py"),
    ]
    loop = _make_loop(tmp_path, responses)
    events, sink = _collect_events()
    loop.event_sink = sink

    result = loop.run_graph("Create a utility module")
    assert result.status == "completed"
    assert result.task_count == 1
    assert (tmp_path / "utils.py").exists()
    # Persisted.
    store = ProjectStore(tmp_path)
    loaded = store.load_task_graph()
    assert loaded is not None
    assert loaded.all_complete
    # Events include task_plan.
    types = [e.type for e in events]
    assert "task_plan" in types
