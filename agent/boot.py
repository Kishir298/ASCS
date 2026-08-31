"""Staged startup / boot sequence for A.S.C.S.

Provides a real, non-faked initialization experience: each stage performs an
actual check and reports its own elapsed time. Failures carry concrete recovery
hints instead of a frozen prompt.
"""

from __future__ import annotations

import platform
import sys
import time as _time
from dataclasses import dataclass, field
from typing import Callable

from . import __version__
from .config import AgentConfig, load_config
from .ollama import OllamaClient, OllamaError
from .workspace import Workspace, WorkspaceError

ProgressFn = Callable[[str, str, float], None]
# Progress handler receives (phase, message, elapsed-seconds) per finished stage.


@dataclass
class BootReport:
    """Outcome of the startup sequence."""

    ok: bool = False
    config: AgentConfig | None = None
    client: OllamaClient | None = None
    workspace: Workspace | None = None
    ollama: dict = field(default_factory=dict)  # ensure_ready report
    stages: list[dict] = field(default_factory=list)  # {phase, message, elapsed}
    error: str = ""
    error_phase: str = ""


class _Runner:
    """Small helper tracking stage timings and progress callbacks."""

    def __init__(self, progress: ProgressFn | None, stages: list[dict], started: float) -> None:
        self.progress = progress
        self.stages = stages
        self.started = started
        self._current: dict | None = None

    def begin(self, phase: str, message: str) -> None:
        self._current = {
            "phase": phase,
            "message": message,
            "elapsed": 0.0,
            "_t0": _time.monotonic(),
        }

    def end(self) -> None:
        if self._current is None:
            return
        self._current["elapsed"] = round(_time.monotonic() - self._current["_t0"], 3)
        item = {k: v for k, v in self._current.items() if not k.startswith("_")}
        self.stages.append(item)
        self._current = None
        if self.progress:
            self.progress(item["phase"], item["message"], item["elapsed"])

    def detail(self, message: str) -> None:
        if self._current is not None:
            self._current["message"] = message


def boot(
    *,
    progress: ProgressFn | None = None,
    workspace_path: str | None = None,
    prewarm: bool | None = None,
    **config_overrides: object,
) -> BootReport:
    """Run the startup sequence and return a :class:`BootReport`.

    ``progress(phase, message, elapsed)`` is invoked after each stage; the
    message is short and truthful ("Connecting to Ollama..."), and elapsed
    seconds reflect the stage that just finished.
    """
    from . import tools as _tools_module  # noqa: F401  (registers tools)

    stages: list[dict] = []
    run = _Runner(progress, stages, _time.monotonic())
    out = BootReport()

    # 1. Configuration.
    run.begin("config", "Loading configuration...")
    try:
        cfg = load_config(**config_overrides)
        if workspace_path:
            fields = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
            fields["workspace"] = workspace_path
            cfg = AgentConfig(**fields)
        out.config = cfg
    except ValueError as exc:
        out.ok = False
        out.error = f"Invalid configuration: {exc}"
        out.error_phase = "config"
        return out
    run.end()

    # 2. Python environment.
    run.begin("pyenv", f"Checking Python environment... (Python {platform.python_version()})")
    if sys.version_info < (3, 12):
        out.ok = False
        out.error = (
            f"Python {platform.python_version()} is too old; A.S.C.S. requires "
            f"Python 3.12 or newer (running {sys.executable}).\n"
            "Recovery: install Python 3.12+ and restart A.S.C.S. with it."
        )
        out.error_phase = "pyenv"
        return out
    run.detail(f"Python {platform.python_version()} OK ({sys.executable})")
    run.end()

    # 3. Workspace.
    run.begin("workspace", "Checking workspace...")
    try:
        out.workspace = Workspace(cfg.workspace)
    except WorkspaceError as exc:
        out.ok = False
        out.error = (
            f"Workspace problem: {exc}\n"
            "Recovery: point the agent at an existing writable directory "
            "(`risa --workspace <path>`), then start again."
        )
        out.error_phase = "workspace"
        return out
    run.detail(f"Workspace ready: {cfg.workspace}")
    run.end()

    # 4. Ollama connectivity.
    run.begin("ollama", f"Connecting to Ollama at {cfg.ollama_base_url}...")
    client = OllamaClient(
        base_url=cfg.ollama_base_url,
        model=cfg.model,
        request_timeout=cfg.request_timeout,
        keep_alive=cfg.keep_alive,
    )
    out.client = client
    try:
        report = client.ensure_ready(
            check_timeout=8,
            warm_timeout=cfg.request_timeout,
            prewarm=False,
        )
    except OllamaError as exc:
        out.ok = False
        out.error = (
            f"Cannot reach Ollama at {cfg.ollama_base_url}: {exc}\n"
            "Recovery: start it with `ollama serve` (or run the Ollama app), "
            "then start A.S.C.S. again."
        )
        out.error_phase = "ollama"
        return out
    out.ollama = report
    run.detail(f"Ollama {report.get('version') or 'connected'} at {cfg.ollama_base_url}")
    run.end()

    # 5. Model availability.
    run.begin("model", f"Checking configured model {cfg.model!r}...")
    if not report["available"]:
        installed = ", ".join(report["installed"]) if report["installed"] else "(none installed)"
        out.ok = False
        out.error = (
            f"Model {cfg.model!r} is not installed on the Ollama server.\n"
            f"Installed models: {installed}\n"
            f"Recovery: run `ollama pull {cfg.model}`, or select an installed "
            "model with OLLAMA_MODEL / --model / the web UI."
        )
        out.error_phase = "model"
        return out
    run.detail(f"Model {cfg.model!r} available; keep_alive={cfg.keep_alive or 'default'}")
    run.end()

    # 6. Warm the model. Slow first load is expected, not an error.
    warm = prewarm if prewarm is not None else cfg.prewarm
    if warm:
        run.begin(
            "warm",
            f"Warming model {cfg.model!r}... (first load of a local model can take a while)",
        )
        try:
            report = client.ensure_ready(
                check_timeout=8, warm_timeout=cfg.request_timeout, prewarm=True
            )
            out.ollama = report
        except OllamaError as exc:
            out.ok = False
            out.error = f"Model warm-up failed: {exc}"
            out.error_phase = "warm"
            return out
        run.detail("Model warmed")
        run.end()

    # 7. Tools.
    run.begin("tools", "Loading agent tools...")
    from .tools import TOOL_SPECS

    run.detail(f"{len(TOOL_SPECS)} tools loaded")
    run.end()

    # 8. Environment: verify the workspace is writable (a real startup sanity
    #    check) and that the test runner is importable.
    run.begin("env", "Running startup checks...")
    from pathlib import Path as _Path

    ws_dir = _Path(cfg.workspace)
    try:
        probe = ws_dir / ".ascs_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        out.ok = False
        out.error = (
            f"Workspace is not writable: {cfg.workspace}: {exc}\n"
            "Recovery: grant write permission to the workspace directory, then start again."
        )
        out.error_phase = "env"
        return out
    try:
        import pytest  # noqa: F401
    except ImportError:
        run.detail("Writable workspace; pytest not installed (tests will be skipped)")
        run.end()
    else:
        run.detail("Writable workspace; pytest available")
        run.end()

    out.ok = True
    out.stages = stages
    return out


def boot_error_message(report: BootReport) -> str:
    """Return a printable recovery message for a failed boot."""
    if report.error:
        return report.error
    return f"Startup did not complete (phase: {report.error_phase or 'unknown'})."


def print_boot(report: BootReport) -> None:
    """Render a BootReport to stdout with timings."""
    print("A.S.C.S. starting...")
    print(f"A.S.C.S. v{__version__} — A Smart Coding System")
    print(
        f"Python {platform.python_version()} ({platform.system()} {platform.machine()})"
    )
    if report.config is not None:
        print(f"Workspace: {report.config.workspace}")
        print(f"Mode: {report.config.mode}  Model: {report.config.model}")
    for st in report.stages:
        elapsed = st.get("elapsed")
        stamp = (
            f"[{elapsed:6.2f}s]" if isinstance(elapsed, (int, float)) and elapsed > 0 else "[     ]"
        )
        print(f"{stamp} {st['message']}")
    if report.ok:
        print("A.S.C.S. ready.")
    else:
        print(boot_error_message(report))
    if sys.stdout and hasattr(sys.stdout, "flush"):
        sys.stdout.flush()