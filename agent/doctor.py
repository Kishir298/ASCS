"""A.S.C.S. diagnostics (``risa doctor``).

Performs read-only environment checks and reports a PASS/WARN/FAIL status for
each, together with an actionable message. Nothing here modifies the workspace,
the Ollama server, or any project state.
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import AgentConfig, load_config
from .context import (DEFAULT_INDEX_FILE, DEFAULT_STATE_DIR, ContextError,
                      ProjectIndex)
from .ollama import OllamaClient
from .workspace import Workspace, WorkspaceError

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one diagnostic check."""

    name: str
    status: str = PASS
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAIL


@dataclass
class DoctorReport:
    """All results for one ``risa doctor`` run."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def warned(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.failed


def _git_available() -> bool:
    try:
        return shutil.which("git") is not None
    except (OSError, ValueError):
        return False


def _pytest_available() -> bool:
    try:
        import pytest  # noqa: F401

        return True
    except ImportError:
        return False


def _check_python() -> CheckResult:
    version = platform.python_version()
    if sys.version_info < (3, 12):
        return CheckResult(
            "python",
            FAIL,
            f"Python {version} is too old; A.S.C.S. requires Python 3.12+ "
            f"({sys.executable}).",
        )
    return CheckResult("python", PASS, f"Python {version} ({sys.executable})")


def _check_install() -> CheckResult:
    try:
        from .workspace import Workspace as _WS  # exercise import chain

        assert _WS is not None
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("install", FAIL, f"A.S.C.S. import failed: {exc}")
    return CheckResult("install", PASS, f"A.S.C.S. v{__version__} importable")


def _check_config(cfg: AgentConfig) -> CheckResult:
    if cfg.mode.upper() not in ("PLAN", "BUILD", "AUTO", "SAFE"):
        return CheckResult("config", FAIL, f"Unsupported mode: {cfg.mode!r}")
    return CheckResult("config", PASS, f"Mode {cfg.mode} / model {cfg.model}")


def _check_workspace(cfg: AgentConfig) -> CheckResult:
    ws = cfg.workspace
    if not ws.exists():
        return CheckResult("workspace", FAIL, f"Workspace does not exist: {ws}")
    if not ws.is_dir():
        return CheckResult("workspace", FAIL, f"Workspace is not a directory: {ws}")
    try:
        Workspace(ws)
    except WorkspaceError as exc:
        return CheckResult("workspace", FAIL, str(exc))
    return CheckResult("workspace", PASS, f"Workspace ready: {ws}")


def _check_ollama(cfg: AgentConfig) -> CheckResult:
    client = OllamaClient(
        base_url=cfg.ollama_base_url,
        model=cfg.model,
        request_timeout=cfg.request_timeout,
        keep_alive=cfg.keep_alive,
    )
    try:
        reachable = client.check_connectivity(timeout=5)
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        return CheckResult(
            "ollama",
            FAIL,
            f"Cannot reach Ollama at {cfg.ollama_base_url}: {exc}. "
            "Recovery: run `ollama serve` (or start the Ollama app).",
        )
    if not reachable:
        return CheckResult(
            "ollama",
            FAIL,
            f"Ollama at {cfg.ollama_base_url} did not answer the connectivity probe. "
            "Recovery: run `ollama serve`.",
        )
    return CheckResult("ollama", PASS, f"Ollama reachable at {cfg.ollama_base_url}")


def _check_model(cfg: AgentConfig) -> CheckResult:
    client = OllamaClient(
        base_url=cfg.ollama_base_url, model=cfg.model, request_timeout=cfg.request_timeout
    )
    try:
        installed = client.list_models(timeout=10)
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        return CheckResult(
            "model", FAIL, f"Could not list models from {cfg.ollama_base_url}: {exc}"
        )
    if cfg.model not in installed:
        found = ", ".join(installed) if installed else "(none installed)"
        return CheckResult(
            "model",
            FAIL,
            f"Model {cfg.model!r} is not installed. Installed: {found}. "
            f"Recovery: run `ollama pull {cfg.model}`.",
        )
    return CheckResult("model", PASS, f"Model {cfg.model!r} installed")


def _check_model_query(cfg: AgentConfig) -> CheckResult:
    """Verify the model is actually queryable, not merely installed.

    Sends a tiny prompt and confirms a non-empty response comes back. A slow
    local model can time out on first load, so a query failure is reported as a
    WARN (non-fatal) rather than a hard FAIL.
    """
    client = OllamaClient(
        base_url=cfg.ollama_base_url,
        model=cfg.model,
        request_timeout=cfg.request_timeout,
        keep_alive=cfg.keep_alive,
    )
    try:
        response = client.chat(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            format=None,
            timeout=max(20, cfg.request_timeout),
        )
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        return CheckResult(
            "model_query",
            WARN,
            f"Model {cfg.model!r} is installed but did not respond to a probe: {exc}. "
            "Recovery: confirm the model serves requests (e.g. `ollama run "
            f"{cfg.model}`) and that keep_alive/request timeouts are sane.",
        )
    if not (response or "").strip():
        return CheckResult(
            "model_query", WARN, f"Model {cfg.model!r} returned an empty reply."
        )
    return CheckResult(
        "model_query", PASS, f"Model {cfg.model!r} responds to a generation probe"
    )


def _check_task_engine(cfg: AgentConfig) -> CheckResult:
    """Verify the task engine (planner + executor + task graph) is importable
    and the graph model round-trips through persistence losslessly."""
    try:
        from .executor import TaskExecutor  # noqa: F401
        from .planner import plan_objective  # noqa: F401
        from .tasks import Task, TaskGraph, build_graph_from_specs
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("task_engine", FAIL, f"Task engine import failed: {exc}")
    try:
        graph = build_graph_from_specs(
            [
                {"id": "T1", "title": "A"},
                {"id": "T2", "title": "B", "dependencies": ["T1"]},
            ]
        )
        round_trip = TaskGraph.from_dict(graph.to_dict())
        assert len(round_trip) == 2
        assert round_trip.task("T2").dependencies == ["T1"]
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult("task_engine", FAIL, f"Task graph model failed: {exc}")
    return CheckResult("task_engine", PASS, "Task engine modules healthy; graph round-trips")


def _check_tools() -> CheckResult:
    from .tools import TOOL_SPECS

    if not TOOL_SPECS:
        return CheckResult("tools", FAIL, "No tools registered.")
    return CheckResult("tools", PASS, f"{len(TOOL_SPECS)} tools registered")


def _check_context_index(cfg: AgentConfig) -> CheckResult:
    index_path = cfg.workspace / DEFAULT_STATE_DIR / DEFAULT_INDEX_FILE
    if not index_path.exists():
        return CheckResult(
            "context",
            WARN,
            f"No context index at {index_path}; the agent will build one on first use.",
        )
    try:
        idx = ProjectIndex(cfg.workspace)
        count = len(idx.records)
    except (ContextError, OSError, ValueError) as exc:
        return CheckResult("context", FAIL, f"Cannot load context index: {exc}")
    return CheckResult("context", PASS, f"Context index healthy ({count} records)")


def _check_project(cfg: AgentConfig) -> CheckResult:
    from .project import ProjectStore

    try:
        store = ProjectStore(cfg.workspace)
        manifest = store.load_manifest()
        languages = ", ".join(
            manifest.languages[:4] if manifest else []
        )
        description = (
            f"{len(store.index.records)} files indexed"
            if store.index.records
            else "no files indexed yet"
        )
        if manifest:
            return CheckResult(
                "project", PASS,
                f"Manifest for {manifest.name!r} ({languages}); {description}.",
            )
        return CheckResult(
            "project", WARN,
            f"No manifest yet; run a task with this workspace to discover "
            f"languages, frameworks and tests ({description}).",
        )
    except (ContextError, OSError) as exc:
        return CheckResult("project", FAIL, f"Cannot open project store: {exc}")


def _check_git(cfg: AgentConfig) -> CheckResult:
    if not _git_available():
        return CheckResult("git", WARN, "git executable not found on PATH.")
    git_dir = cfg.workspace / ".git"
    if not git_dir.exists():
        return CheckResult("git", WARN, "Workspace is not a git repository.")
    return CheckResult("git", PASS, "git available; workspace is a repository")


def _check_runtime_platform() -> CheckResult:
    """Runtime is Windows-only; dev testing is cross-platform.

    The agent's shell workflow (cmd.exe, ``python``/``pip`` on PATH) is
    Windows-first. Dev testing via ``pytest`` is intentionally cross-platform
    (``tools._python_fallback_command`` handles ``python`` -> ``python3``),
    but the full runtime (Ollama + risa TUI/Web) is only supported on Windows.
    """
    import os as _os

    if _os.name == "nt":
        return CheckResult("platform", PASS, "Windows runtime (supported)")
    return CheckResult(
        "platform",
        WARN,
        "Non-Windows host — dev testing (pytest) is cross-platform, but the "
        "full ASCS runtime (Ollama/TUI/Web) is Windows-only. "
        "Tool execution falls back ``python`` -> ``python3`` for dev.",
    )


def _check_tests(cfg: AgentConfig) -> CheckResult:
    if not _pytest_available():
        return CheckResult("tests", WARN, "pytest is not installed.")
    return CheckResult("tests", PASS, "pytest available")


def doctor(*, workspace: str | Path | None = None, **config_overrides) -> DoctorReport:
    """Run the read-only diagnostic suite for a workspace.

    A config error is reported as a ``config`` FAIL, but the remaining checks
    still run against a best-effort configuration so the report surfaces every
    problem at once (e.g. a missing workspace is still diagnosed).
    """
    results: list[CheckResult] = []

    def _best_effort_cfg() -> AgentConfig:
        fields = dict(config_overrides)
        if workspace is not None:
            fields["workspace"] = workspace
        try:
            return AgentConfig(**fields)
        except TypeError:  # pragma: no cover - caller always passes valid fields
            return AgentConfig()

    try:
        cfg = load_config(**config_overrides)
        if workspace is not None:
            fields = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
            fields["workspace"] = Path(workspace).expanduser()
            cfg = AgentConfig(**fields)
        results.append(_check_config(cfg))
    except ValueError as exc:
        results.append(CheckResult("config", FAIL, f"Invalid configuration: {exc}"))
        cfg = _best_effort_cfg()

    results.append(_check_runtime_platform())
    results.append(_check_python())
    results.append(_check_install())
    results.append(_check_workspace(cfg))
    results.append(_check_ollama(cfg))
    results.append(_check_model(cfg))
    results.append(_check_model_query(cfg))
    results.append(_check_tools())
    results.append(_check_context_index(cfg))
    results.append(_check_project(cfg))
    results.append(_check_git(cfg))
    results.append(_check_tests(cfg))
    results.append(_check_task_engine(cfg))
    return DoctorReport(results)


def print_doctor(report: DoctorReport, *, progress: bool = False) -> None:
    """Render a DoctorReport to stdout in the ``risa doctor`` style."""
    print("A.S.C.S. doctor")
    print(f"  A.S.C.S. v{__version__} on Python {platform.python_version()}")
    width = max(len(r.name) for r in report.results) if report.results else 8
    for result in report.results:
        print(f"  [{result.status:4}] {result.name:<{width}}  {result.message}")
    if report.ok and not report.warned:
        print("  All checks passed.")
    elif report.ok:
        print(f"  {len(report.warned)} warning(s); no failures.")
    else:
        print(f"  {len(report.failed)} failure(s); {len(report.warned)} warning(s).")


__all__ = [
    "CheckResult",
    "DoctorReport",
    "doctor",
    "print_doctor",
    "PASS",
    "WARN",
    "FAIL",
]