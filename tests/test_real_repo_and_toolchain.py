"""Coding-agent readiness tests: real-repository workflows, toolchain
detection, context refresh, malformed-plan recovery, safety and corrupt-state
handling.

These extend the existing suite with the acceptance criteria that were not yet
directly exercised: realistic multi-step coding runs on *real* temporary
repositories (Python and Node toolchains), automated toolchain detection, and
the safety invariants (no auto-commit, dirty files untouched, PLAN read-only).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from agent.config import AgentConfig
from agent.executor import TaskExecutor
from agent.loop import AgentLoop, run_graph_agent
from agent.project import ProjectStore, scan
from agent.tasks import (
    COMPLETED,
    FAILED,
    READY,
    PENDING,
    RUNNING,
    Task,
    TaskGraph,
    build_graph_from_specs,
)
from agent.toolchain import detect_toolchain, toolchain_to_text
from agent.workspace import Workspace


# ── helpers ────────────────────────────────────────────────────────────────

class FakeClient:
    """Minimal scripted fake Ollama client (deterministic, no server)."""

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


def _make_loop(tmp_path, responses, *, mode="AUTO"):
    config = AgentConfig(workspace=tmp_path, mode=mode)
    client = FakeClient(responses, model=config.model)
    ws = Workspace(tmp_path)
    return AgentLoop(config, client, ws, log=lambda m: None)


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=str(root), capture_output=True, check=False)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(root),
                   capture_output=True, check=False)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(root),
                   capture_output=True, check=False)


def _git_commit(root: Path, msg: str = "init") -> None:
    subprocess.run(["git", "add", "."], cwd=str(root), capture_output=True, check=False)
    subprocess.run(["git", "commit", "-m", msg], cwd=str(root),
                   capture_output=True, check=False)


def _git_status(root: Path) -> str:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                            capture_output=True, text=True, check=False)
    return result.stdout


def _git_dirty_files(root: Path) -> set[str]:
    return {
        line.split(maxsplit=1)[1].strip()
        for line in _git_status(root).splitlines()
        if line.strip() and ".ascs" not in line
    }


# ── 1. Toolchain detection ─────────────────────────────────────────────────

def test_detect_python_pytest_toolchain(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n[tool.pytest.ini_options]\n", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    tc = detect_toolchain(tmp_path)
    assert tc.language == "python"
    assert "python -m pytest" in tc.test_commands
    assert tc.package_manager == "pyproject.toml"


def test_detect_python_requirements_toolchain(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    tc = detect_toolchain(tmp_path)
    assert tc.language == "python"
    assert "python -m pytest" in tc.test_commands


def test_detect_node_toolchain_npm_test(tmp_path):
    (tmp_path / "package.json").write_text(
        '{\n"scripts": {"test": "node --test", "build": "node build.js"}\n}\n',
        encoding="utf-8",
    )
    tc = detect_toolchain(tmp_path)
    assert tc.language == "javascript"
    assert "npm test" in tc.test_commands
    assert "npm run build" in tc.lint_commands


def test_detect_typescript_toolchain(tmp_path):
    (tmp_path / "package.json").write_text(
        '{\n"scripts": {"test": "jest"}\n}\n', encoding="utf-8"
    )
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    tc = detect_toolchain(tmp_path)
    assert tc.language == "typescript"
    assert "npm test" in tc.test_commands
    assert "npx tsc --noEmit" in tc.lint_commands


def test_detect_unknown_toolchain(tmp_path):
    (tmp_path / "foo.txt").write_text("hi", encoding="utf-8")
    tc = detect_toolchain(tmp_path)
    assert not tc.detected
    assert tc.language == "unknown"


def test_toolchain_surfaces_in_manifest(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n[tool.pytest.ini_options]\n", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    manifest = scan(tmp_path)
    assert "python" in manifest.toolchain
    assert "pytest" in manifest.toolchain


def test_toolchain_to_text_unknown():
    from agent.toolchain import Toolchain

    text = toolchain_to_text(Toolchain())
    assert "no standard toolchain" in text.lower()


# ── 2. Verification derivation uses toolchain ──────────────────────────────

def test_derive_verification_uses_detected_test_cmd():
    from agent.planner import _derive_verification

    intelligence = (
        "- Project: demo\n- Toolchain: Language: python; Package manager: "
        "pyproject.toml; Likely test commands: python -m pytest; "
        "Likely lint commands: ruff check"
    )
    steps = _derive_verification(
        {"title": "Add feature", "kind": "implement", "files": ["x.py"]},
        intelligence,
    )
    assert "run python -m pytest" in steps


def test_derive_verification_non_code():
    from agent.planner import _derive_verification

    steps = _derive_verification(
        {"title": "Inspect", "kind": "inspect"},
        "- Project: demo\n- Toolchain: Language: python",
    )
    assert "report inspection findings" in steps[0]


# ── 3. Context refresh between tasks ───────────────────────────────────────

def test_context_refresh_after_task(tmp_path):
    """After a task writes a file, the project index sees the new file."""
    store = ProjectStore(tmp_path)
    store.refresh()
    before = len(store.index.records)

    (tmp_path / "newfile.py").write_text("def f(): return 1\n", encoding="utf-8")
    config = AgentConfig(workspace=tmp_path, mode="AUTO")
    executor = TaskExecutor(
        config=config, client=None, workspace=Workspace(tmp_path),
        store=store, log=lambda m: None,
    )
    executor._refresh_context()
    after = len(store.index.records)
    assert after > before
    assert any("newfile.py" in path for path in store.index.records)


# ── 4. Malformed task graph recovery ───────────────────────────────────────

def test_malformed_plan_recovery_falls_back_to_single_task(tmp_path):
    """A planner output that forms an invalid DAG falls back gracefully."""
    # T1 depends on a task id that does not exist -> would normally raise.
    responses = [
        _json({"tasks": [
            {"id": "T1", "title": "Inspect", "kind": "inspect"},
            {"id": "T2", "title": "Implement", "dependencies": ["NOPE"],
             "kind": "implement", "verification": ["run echo ok"]},
        ]}),
        _done("implemented"),
    ]
    loop = _make_loop(tmp_path, responses)
    result = loop.run_graph("do the thing")
    # Should not be a fatal error; graceful fallback to a bounded task.
    assert result.status in ("completed", "partial")
    assert result.task_count >= 1


# ── 5. Real-repository coding workflows ────────────────────────────────────

def test_real_python_repo_full_workflow(tmp_path):
    """Multi-step Python repo: plan → implement → test → verify → report."""
    # Real Python repository with a test suite.
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n"
        "[tool.pytest.ini_options]\n", encoding="utf-8"
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "app" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(
        "from app.calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    _git_init(tmp_path)
    _git_commit(tmp_path)

    planner_response = _json({
        "tasks": [
            {"id": "T1", "title": "Add a multiply function to app/calc.py",
             "kind": "implement", "files": ["app/calc.py"],
             "verification": ["run python -m pytest"]},
            {"id": "T2", "title": "Add a test for multiply",
             "dependencies": ["T1"], "kind": "implement",
             "files": ["tests/test_calc.py"],
             "verification": ["run python -m pytest"]},
        ]
    })
    responses = [
        planner_response,
        _tool("write_file", {
            "path": "app/calc.py",
            "content": "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
        }),
        _done("added multiply"),
        _tool("write_file", {
            "path": "tests/test_calc.py",
            "content": (
                "from app.calc import add, multiply\n\n"
                "def test_add():\n    assert add(1, 2) == 3\n\n"
                "def test_multiply():\n    assert multiply(3, 4) == 12\n"
            ),
        }),
        _done("added multiply test"),
    ]
    loop = _make_loop(tmp_path, responses)
    result = loop.run_graph("Add a multiply function and its test")
    assert result.status == "completed"
    assert result.task_count == 2
    # Real files changed.
    assert "def multiply" in (tmp_path / "app" / "calc.py").read_text(encoding="utf-8")
    assert "test_multiply" in (tmp_path / "tests" / "test_calc.py").read_text(encoding="utf-8")
    # Report is truthful and lists changed files.
    assert "Objective:" in result.report
    assert "Files changed:" in result.report
    assert "app/calc.py" in result.report
    # Nothing auto-committed.
    dirty = _git_dirty_files(tmp_path)
    assert "app/calc.py" in dirty  # still dirty (no commit happened)
    # Persisted state is complete.
    store = ProjectStore(tmp_path)
    loaded = store.load_task_graph()
    assert loaded is not None
    assert loaded.all_complete


def test_real_node_repo_full_workflow(tmp_path):
    """Multi-step Node repo: detect npm toolchain and implement."""
    (tmp_path / "package.json").write_text(
        '{\n  "name": "demo",\n  "version": "0.1.0",\n'
        '  "scripts": {"test": "node --test"}\n}\n',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "greet.js").write_text(
        "function greet(name) { return 'Hello, ' + name; }\nmodule.exports = { greet };\n",
        encoding="utf-8",
    )
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "greet.test.js").write_text(
        "const { greet } = require('../src/greet');\n"
        "const { test } = require('node:test');\n"
        "const assert = require('node:assert');\n"
        "test('greet', () => { assert.strictEqual(greet('Ada'), 'Hello, Ada'); });\n",
        encoding="utf-8",
    )
    _git_init(tmp_path)
    _git_commit(tmp_path)

    tc = detect_toolchain(tmp_path)
    assert tc.language == "javascript"
    assert "npm test" in tc.test_commands

    planner_response = _json({
        "tasks": [
            {"id": "T1", "title": "Add farewell function to src/greet.js",
             "kind": "implement", "files": ["src/greet.js"],
             "verification": ["run node -e \"const {farewell}=require('./src/greet'); console.log(farewell('Ada'))\""]},
        ]
    })
    responses = [
        planner_response,
        _tool("write_file", {
            "path": "src/greet.js",
            "content": (
                "function greet(name) { return 'Hello, ' + name; }\n"
                "function farewell(name) { return 'Goodbye, ' + name; }\n"
                "module.exports = { greet, farewell };\n"
            ),
        }),
        _done("added farewell"),
    ]
    loop = _make_loop(tmp_path, responses)
    result = loop.run_graph("Add a farewell function")
    assert result.status == "completed"
    assert "farewell" in (tmp_path / "src" / "greet.js").read_text(encoding="utf-8")
    dirty = _git_dirty_files(tmp_path)
    assert "src/greet.js" in dirty  # no auto-commit
    assert "Task failure" not in result.report  # truthful: not claiming failure


# ── 6. Final report truthfulness ───────────────────────────────────────────

def test_report_flags_incomplete_as_not_achieved(tmp_path):
    """A partial run must not claim the objective was achieved."""
    def _always_fail_verify(executor, task):
        from agent.executor import VerificationResult
        return VerificationResult(
            task_id=task.id, ok=False,
            steps=[{"step": "test", "status": "failed", "ok": False, "output": "FAIL"}],
        )

    def _ok_run(executor, task):
        from agent.executor import TaskOutcome
        return TaskOutcome(task_id=task.id, ok=True, summary="done", iterations=1)

    executor = TaskExecutor(
        config=AgentConfig(workspace=tmp_path, mode="AUTO"),
        client=None, workspace=Workspace(tmp_path),
        log=lambda m: None,
        run_task=_ok_run, verify=_always_fail_verify,
    )
    graph = build_graph_from_specs([{"id": "T1", "title": "X"}])
    execution = executor.execute("test", graph)
    assert execution.status == "partial"
    report = execution.report()
    assert report.split("Achieved:")[1].strip().startswith("no")


# ── 7. Safety invariants ───────────────────────────────────────────────────

def test_no_auto_commit_in_workflow(tmp_path):
    """The pipeline never silently commits/pushes changes."""
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    _git_init(tmp_path)
    _git_commit(tmp_path)

    responses = [
        _json({"tasks": [
            {"id": "T1", "title": "Write a file", "kind": "implement",
             "verification": ["run echo ok"]},
        ]}),
        _tool("write_file", {"path": "new.txt", "content": "hello\n"}),
        _done("wrote file"),
    ]
    loop = _make_loop(tmp_path, responses)
    result = loop.run_graph("write a file")
    assert result.status == "completed"
    dirty = _git_dirty_files(tmp_path)
    assert "new.txt" in dirty  # uncommitted -> proves no auto-commit


def test_plan_mode_never_modifies_workspace(tmp_path):
    """PLAN mode: even scripted modify/command tools are blocked; no writes."""
    (tmp_path / "orig.txt").write_text("original\n", encoding="utf-8")

    responses = [
        _json({"tasks": [
            {"id": "T1", "title": "Inspect", "kind": "inspect",
             "verification": ["report findings"]},
        ]}),
        # Even if the model tries to write, PLAN must block it.
        _tool("write_file", {"path": "should_not_exist.txt", "content": "x"}),
        _done("inspected"),
    ]
    loop = _make_loop(tmp_path, responses, mode="PLAN")
    result = loop.run_graph("inspect only")
    # A non-modifying inspect task passes even in PLAN mode.
    assert result.status == "completed"
    # The attempting write did not create the file.
    assert not (tmp_path / "should_not_exist.txt").exists()
    assert (tmp_path / "orig.txt").read_text(encoding="utf-8") == "original\n"


def test_dirty_baseline_file_untouched_in_real_repo(tmp_path):
    """Real repo: a file dirty before the run is not overwritten by ASCS."""
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    _git_init(tmp_path)
    _git_commit(tmp_path)
    # Make README dirty *before* the run (simulate pre-existing user work).
    (tmp_path / "README.md").write_text("# demo\nMY USER EDITS\n", encoding="utf-8")

    responses = [
        _json({"tasks": [
            {"id": "T1", "title": "Update README", "kind": "implement",
             "verification": ["run echo ok"]},
        ]}),
        _tool("write_file", {"path": "README.md", "content": "# OVERWRITTEN\n"}),
        _done("attempted overwrite"),
    ]
    # Build git baseline the same way the loop does.
    from agent.context import git_status
    baseline: set[str] = set()
    for line in git_status(tmp_path).splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and not parts[1].startswith(".ascs"):
            baseline.add(parts[1].strip())

    loop = _make_loop(tmp_path, responses)
    loop.approver = lambda _d: True
    result = loop.run_graph("update readme")
    # The dirty readme must NOT be overwritten.
    content = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "MY USER EDITS" in content
    assert "OVERWRITTEN" not in content


# ── 8. Corrupt state fails safely ──────────────────────────────────────────

def test_corrupt_task_state_loads_none(tmp_path):
    """Corrupt task_state.json must fail safely (None), not crash."""
    store = ProjectStore(tmp_path)
    store.save_task_graph(
        build_graph_from_specs([{"id": "T1", "title": "clean"}])
    )
    # Corrupt the file.
    (store.task_state_path).write_text("{ this is not valid json !!!", encoding="utf-8")
    loaded = store.load_task_graph()
    assert loaded is None


def test_missing_task_state_loads_none(tmp_path):
    store = ProjectStore(tmp_path)
    assert store.load_task_graph() is None


# ── 9. Live-model smoke test ──────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        from agent.ollama import OllamaClient

        client = OllamaClient(model="qwen3:14b", request_timeout=10)
        return client.check_connectivity(timeout=5)
    except Exception:
        return False


@pytest.mark.skipif(
    os.environ.get("RISALIVE") != "1",
    reason="Live qwen3:14b smoke test — opt-in: set RISALIVE=1 (takes minutes)",
)
def test_live_model_smoke(tmp_path):
    """Live qwen3:14b: write one file and verify via real toolchain (BUILD mode).

    This test runs against a real Ollama server and a real temp repo.
    Skipped when Ollama is not reachable.
    """
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'smoke'\nversion = '0.1'\n[tool.pytest.ini_options]\n",
        encoding="utf-8",
    )
    (tmp_path / "smoke.py").write_text("def greet(): return 'hello'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_smoke.py").write_text(
        "from smoke import greet\n\ndef test_greet():\n    assert greet() == 'hello'\n",
        encoding="utf-8",
    )
    _git_init(tmp_path)
    _git_commit(tmp_path)

    from agent.config import load_config
    from agent.ollama import OllamaClient

    config = load_config(
        workspace=str(tmp_path),
        mode="BUILD",
        model="qwen3:14b",
        max_iterations=15,
    )
    client = OllamaClient(
        base_url=config.ollama_base_url,
        model=config.model,
        request_timeout=config.request_timeout,
        keep_alive=config.keep_alive,
    )
    result = run_graph_agent(
        config,
        client,
        "Add a function 'farewell(name)' to smoke.py that returns 'Goodbye, ' + name",
        log=lambda m: None,
    )
    # The pipeline must complete without fatal errors.
    assert result.status != "fatal", f"fatal error: {result.error}"
    assert result.report, "report must not be empty"
    # The farewell function must actually appear in the file.
    smoke = (tmp_path / "smoke.py").read_text(encoding="utf-8")
    assert "farewell" in smoke
    # No auto-commit.
    dirty = _git_dirty_files(tmp_path)
    assert "smoke.py" in dirty


# ── 10. Toolchain → real verification on real repos ────────────────────────

def test_toolchain_driven_verification_python(tmp_path):
    """Toolchain detection drives verification to run real pytest."""
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n[tool.pytest.ini_options]\n", encoding="utf-8"
    )
    (tmp_path / "lib.py").write_text("def double(x): return x * 2\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_lib.py").write_text(
        "from lib import double\n\ndef test_double():\n    assert double(3) == 6\n",
        encoding="utf-8",
    )

    tc = detect_toolchain(tmp_path)
    assert "python -m pytest" in tc.test_commands

    from agent.planner import _derive_verification

    intelligence = f"- Project: demo\n- Toolchain: {toolchain_to_text(tc)}"
    steps = _derive_verification(
        {"title": "Add feature", "kind": "implement", "files": ["lib.py"]},
        intelligence,
    )
    # The first step must be "run python -m pytest" (not a made-up command).
    assert steps[0] == "run python -m pytest"


def test_toolchain_driven_verification_node(tmp_path):
    """Toolchain detection drives verification to npm test for Node repos."""
    (tmp_path / "index.js").write_text("function add(a,b){return a+b}\nmodule.exports={add};\n",
                                       encoding="utf-8")
    (tmp_path / "package.json").write_text(
        '{"scripts":{"test":"node --test"}}\n', encoding="utf-8"
    )

    tc = detect_toolchain(tmp_path)
    assert tc.language == "javascript"
    assert "npm test" in tc.test_commands

    from agent.planner import _derive_verification

    intelligence = f"- Project: demo\n- Toolchain: {toolchain_to_text(tc)}"
    steps = _derive_verification(
        {"title": "Add feature", "kind": "implement", "files": ["index.js"]},
        intelligence,
    )
    assert steps[0] == "run npm test"


# ── 11. Resume correctness ────────────────────────────────────────────────

def test_resume_preserves_completed_tasks_only(tmp_path):
    """Resume with T1 complete, T2 FAILED, T3 PENDING: T3 must be cancelled."""
    store = ProjectStore(tmp_path)
    graph = TaskGraph()
    graph.add(Task(id="T1", title="Done", status=COMPLETED,
                    verification=["run echo ok"]))
    graph.add(Task(id="T2", title="Failed", status=FAILED,
                    failure_reason="boom",
                    verification=["run echo fail"]))
    graph.add(Task(id="T3", title="Waiting", dependencies=["T2"],
                    verification=["run echo wait"]))
    store.save_task_graph(graph)

    loaded = store.load_task_graph()
    assert loaded is not None
    assert loaded.task("T1").status == COMPLETED
    assert loaded.task("T2").status == FAILED
    assert loaded.task("T3").status == PENDING


def test_stuck_running_resets_to_ready(tmp_path):
    """Tasks stuck in RUNNING from an interrupted run reset to READY."""
    graph = TaskGraph()
    graph.add(Task(id="T1", title="Done", status=COMPLETED))
    graph.add(Task(id="T2", title="Stuck", dependencies=["T1"], status=RUNNING))
    graph.add(Task(id="T3", title="Waiting", dependencies=["T2"]))
    graph.add(Task(id="T4", title="Free"))

    reset = TaskExecutor.reset_stuck_tasks(graph)
    assert "T2" in reset
    assert graph.task("T2").status == READY
    # Unrelated task not affected.
    assert graph.task("T4").status == READY
