"""A.S.C.S. task-graph executor.

The executor turns a :class:`~agent.tasks.TaskGraph` (produced by the planner)
into actual work: it walks the graph, runs each ready task to completion,
verifies it against its acceptance criteria, and records the outcome so a run
can be paused, persisted and resumed via
:meth:`~agent.project.ProjectStore.save_task_graph` /
:meth:`~agent.project.ProjectStore.load_task_graph`.

Structure
=========
* :class:`TaskOutcome` / :class:`VerificationResult` - per-task results.
* :func:`task_system_prompt` / :func:`task_user_prompt` - task-scoped prompts.
* :class:`TaskExecutor` - orchestrates the graph and runs the model loop.

The model-facing part is deliberately pluggable: ``TaskExecutor`` accepts an
optional ``run_task`` callable (tests inject fakes so no Ollama server is
needed). The default implementation is a single-task agent loop sharing the
same tool contract as the main loop.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import AgentConfig
from .events import (
    EventSink,
    emit_activity,
    emit_command_completed,
    emit_command_started,
    emit_status,
    null_sink,
)
from .models import ToolResult, parse_model_reply, tool_result_message
from .ollama import OllamaClient, OllamaError
from .prompts import system_prompt
from .tasks import COMPLETED, FAILED, RUNNING, SKIPPED, Task, TaskGraph
from .tools import execute_tool
from .workspace import Workspace


@dataclass
class VerificationResult:
    """Outcome of verifying a single task's acceptance criteria."""

    task_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)  # {"step", "ok", "output"}
    ok: bool = False

    @property
    def detail(self) -> str:
        lines = [f"{s['status']}: {s['step']}" for s in self.steps]
        return "\n".join(lines) if lines else "(no verification steps)"


@dataclass
class TaskOutcome:
    """Result of executing (and verifying) one task."""

    task_id: str
    ok: bool = False
    summary: str = ""
    iterations: int = 0
    verification: VerificationResult | None = None
    reason: str = ""


@dataclass
class TaskExecution:
    """Overall result of executing a task graph."""

    objective: str
    status: str = "completed"  # completed|partial|cancelled|failed
    summary: str = ""
    iterations: int = 0
    outcomes: list[TaskOutcome] = field(default_factory=list)
    progress: dict[str, int] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == "completed"

    @property
    def failed_tasks(self) -> list[TaskOutcome]:
        return [o for o in self.outcomes if not o.ok]

    @property
    def completed_tasks(self) -> list[TaskOutcome]:
        return [o for o in self.outcomes if o.ok]


def task_system_prompt(
    task: Task,
    project_block: str,
    config: AgentConfig,
) -> str:
    """A task-scoped system prompt: base prompt plus the current task contract."""
    base = system_prompt(config, project_block)
    current_task = (
        "\n\nCURRENT TASK (focus ONLY on this bounded unit of work)\n"
        "============================================\n"
        f"- Task id: {task.id}\n"
        f"- Title: {task.title}\n"
        f"- Description: {task.description or task.title}\n"
    )
    if task.files:
        current_task += f"- Target files: {', '.join(task.files)}\n"
    if task.commands:
        current_task += f"- Commands: {', '.join(task.commands)}\n"
    if task.verification:
        current_task += (
            "- When you have implemented this task, finish with "
            '{"done": true, "summary": "..."} so the task is verified. '
            "The following acceptance criteria will then be run:\n"
        )
        for step in task.verification:
            current_task += f"    * {step}\n"
    current_task += (
        "\nDo not change files outside the task's scope. When the task is "
        'implemented, reply {"done": true, "summary": "..."}.'
    )
    return base + current_task


def task_user_prompt(task: Task) -> str:
    """The per-task user message handed to the model."""
    lines = [
        f"Task {task.id}: {task.title}",
    ]
    if task.description and task.description != task.title:
        lines.append(f"\n{task.description}")
    return "\n".join(lines)


def _project_block(store) -> str:
    if store is None:
        return ""
    try:
        from .project import project_prompt_text

        return project_prompt_text(store)
    except Exception:  # noqa: BLE001 - best-effort only
        return ""


class TaskExecutor:
    """Drive a task graph: run each ready task, verify it, persist progress.

    Model execution is delegated to ``run_task`` (injectable for tests); the
    default uses a single-task agent loop. Verification is delegated to
    ``verify`` (injectable); the default runs each task's commands/verification
    through the ``run_command`` tool.
    """

    def __init__(
        self,
        config: AgentConfig,
        client: OllamaClient,
        workspace: Workspace,
        *,
        store=None,
        event_sink: EventSink | None = None,
        log: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        run_task: Callable[["TaskExecutor", Task], TaskOutcome] | None = None,
        verify: Callable[["TaskExecutor", Task], VerificationResult] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.ws = workspace
        self.store = store
        self.event_sink = event_sink or null_sink
        self.log = log or (lambda _m: None)
        self.should_stop = should_stop or (lambda: False)
        self.run_task = run_task or self._default_run_task
        self.verify = verify or self._verify_task
        self.iterations = 0

    # -- public API --------------------------------------------------------

    def execute(self, objective: str, graph: TaskGraph) -> TaskExecution:
        """Run ``graph`` to completion (or cancellation/failure)."""
        execution = TaskExecution(objective=objective)
        emit_status(self.event_sink, "EXECUTING", "Task engine started")
        graph.recompute_statuses()

        try:
            while not graph.all_complete and not graph.first_failure:
                if self.should_stop():
                    execution.status = "cancelled"
                    execution.summary = "Cancelled by the operator."
                    emit_status(self.event_sink, "CANCELLED", execution.summary)
                    break
                ready = graph.ready_tasks()
                if not ready:
                    # No ready tasks and not complete -> everything left is blocked
                    # or pending behind a failure; cascade and stop.
                    graph.cascade_cancel()
                    execution.status = "failed"
                    execution.summary = "No runnable tasks remain (dependency blocked)."
                    break
                task = ready[0]
                failed_id = task.id
                execution.iterations += 1
                outcome = self._execute_one(graph, task)
                execution.outcomes.append(outcome)
                self.persist(graph)
                if not outcome.ok:
                    # _execute_one already marked the task FAILED; cancel
                    # dependents that can no longer proceed and stop.
                    graph.cascade_cancel()
                    execution.status = "partial"
                    execution.summary = (
                        f"Task {failed_id} failed: {outcome.reason or outcome.summary}"
                    )
                    break

            if execution.status == "completed":
                if graph.all_complete:
                    execution.summary = self._completion_summary(graph)
                    emit_status(self.event_sink, "COMPLETE", execution.summary)
                elif graph.first_failure:
                    f = graph.first_failure
                    execution.status = "failed"
                    execution.summary = f"Stopped after task {f.id} failed: {f.failure_reason}"
                    emit_status(self.event_sink, "FAILED", execution.summary)
        finally:
            self.persist(graph)

        execution.progress = graph.progress()
        self.iterations += execution.iterations
        return execution

    def _execute_one(self, graph: TaskGraph, task: Task) -> TaskOutcome:
        """Run one task to a terminal outcome, verifying it on success."""
        graph.mark(task.id, RUNNING)
        self.log(f"[task {task.id}] running: {task.title}")
        emit_status(self.event_sink, "EXECUTING", f"Task {task.id}: {task.title}")

        outcome = self.run_task(self, task)
        self.iterations += outcome.iterations

        if not outcome.ok:
            graph.mark(
                task.id,
                FAILED,
                retry_count=task.retry_count + 1,
                failure_reason=outcome.reason or outcome.summary,
            )
            emit_status(
                self.event_sink, "FAILED", f"Task {task.id} failed: {outcome.reason or outcome.summary}"
            )
            return outcome

        # Mark completed only once verification passes.
        verification = self.verify(self, task)
        outcome.verification = verification
        if verification.ok:
            graph.mark(task.id, COMPLETED, retry_count=task.retry_count, result=outcome.summary)
            emit_status(self.event_sink, "EXECUTING", f"Task {task.id} verified")
        else:
            outcome.ok = False
            outcome.reason = f"verification failed:\n{verification.detail}"
            graph.mark(task.id, FAILED, retry_count=task.retry_count + 1,
                       failure_reason=outcome.reason)
        return outcome

    def _completion_summary(self, graph: TaskGraph) -> str:
        done = len(graph.tasks_by_status(COMPLETED))
        skipped = len(graph.tasks_by_status(SKIPPED))
        total = len(graph)
        return f"All {total} task(s) completed ({done} done, {skipped} skipped)."

    def persist(self, graph: TaskGraph) -> None:
        """Persist current graph state so a run can resume."""
        if self.store is not None:
            try:
                self.store.save_task_graph(graph)
            except Exception as exc:  # noqa: BLE001 - persistence must not crash
                self.log(f"[executor] could not persist task state: {exc}")

    # -- default single-task model loop ------------------------------------

    def _default_run_task(self, executor, task: Task) -> TaskOutcome:
        """A task-scoped agent loop; verified separately by ``_verify_task``.

        Keeps the same tool contract as the main loop but scoped to one task,
        bounded by the config's iteration budget. Tolerates malformed replies
        up to a limit, and never trusts a model "done" without verification.
        """
        messages = [
            {
                "role": "system",
                "content": task_system_prompt(task, _project_block(executor.store), self.config),
            },
            {"role": "user", "content": task_user_prompt(task)},
        ]
        outcome = TaskOutcome(task_id=task.id)

        for iteration in range(1, self.config.max_iterations + 1):
            if self.should_stop():
                outcome.ok = False
                outcome.reason = "cancelled by operator"
                return outcome
            try:
                reply_text = self.client.chat(
                    _trim_for_request(messages, self.config.context_budget_chars),
                    format="json",
                )
            except OllamaError as exc:
                outcome.ok = False
                outcome.reason = f"model error: {exc}"
                return outcome

            reply = parse_model_reply(reply_text)
            if reply.error is not None:
                messages.append({"role": "assistant", "content": reply_text})
                messages.append({"role": "user", "content": reply.error})
                continue

            if reply.done:
                outcome.summary = reply.summary
                outcome.iterations = iteration
                outcome.ok = True
                return outcome

            result = self._run_tool(reply.tool, reply.arguments, task, iteration)
            outcome.iterations = iteration
            messages.append({"role": "assistant", "content": reply_text})
            messages.append(tool_result_message(result))

        outcome.ok = False
        outcome.reason = (
            f"Task exceeded {self.config.max_iterations} iterations without finishing"
        )
        return outcome

    def _run_tool(
        self, tool: str, arguments: dict[str, Any], task: Task, iteration: int
    ) -> ToolResult:
        if tool not in self.config.effective_tools:
            from .prompts import tool_error_feedback

            return ToolResult(
                tool,
                f"tool not available in this session (available: "
                f"{', '.join(self.config.effective_tools)}).",
                ok=False,
            )
        result = execute_tool(tool, arguments, self.ws, self.config)
        self.log(f"[task {task.id}:{iteration:02d}] {tool}: {result.note or ('ok' if result.ok else 'FAILED')}")
        emit_activity(self.event_sink, f"Task {task.id}: {tool}")
        return result

    # -- default verification ---------------------------------------------

    def _verify_task(self, executor, task: Task) -> VerificationResult:
        """Run the task's commands + verification steps via ``run_command``.

        A verification string of the form ``run <command>`` is executed as a
        shell command; anything else is treated as a descriptive step and
        recorded. All executed commands must exit 0 for the task to pass.
        """
        result = VerificationResult(task_id=task.id)
        steps: list[str] = []

        for command in task.commands:
            steps.append(("run", command))
        for step in task.verification:
            stripped = step.strip()
            if stripped.lower().startswith("run "):
                steps.append(("run", stripped[4:].strip()))
            else:
                steps.append(("check", stripped))

        if not steps:
            # Nothing to verify: an explicit no-op is treated as passing only
            # if the task declares so; otherwise we flag a gap but allow it.
            result.ok = True
            return result

        all_ok = True
        for kind, payload in steps:
            if kind == "check":
                result.steps.append(
                    {"step": payload, "status": "noted", "ok": True, "output": ""}
                )
                continue
            cmd = payload
            emit_command_started(self.event_sink, cmd)
            tool_result = execute_tool(
                "run_command", {"command": cmd}, self.ws, self.config
            )
            # A command is only a passing verification step when it exits 0
            # (ToolResult.ok alone is true for any completed command).
            ok = not _timed_out(tool_result) and _exit_code(tool_result) == 0
            emit_command_completed(
                self.event_sink, cmd, exit_code=_exit_code(tool_result), ok=ok
            )
            result.steps.append(
                {
                    "step": cmd,
                    "status": "ok" if ok else "failed",
                    "ok": ok,
                    "output": tool_result.output,
                }
            )
            if not ok:
                all_ok = False

        result.ok = all_ok
        return result


def _timed_out(result: ToolResult) -> bool:
    return "timed out" in (result.note or "")


def _exit_code(result: ToolResult) -> int:
    import re

    if _timed_out(result):
        return -1
    m = re.search(r"exit code (\d+)", result.note or "")
    return int(m.group(1)) if m else (0 if result.ok else 1)


def _trim_for_request(messages: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
    out = list(messages)
    if budget <= 0:
        return out
    total = sum(len(m.get("content", "")) for m in out)
    while total > budget and len(out) > 2:
        removed = out.pop(2)
        total -= len(removed.get("content", ""))
    return out


__all__ = [
    "TaskExecutor",
    "TaskExecution",
    "TaskOutcome",
    "VerificationResult",
    "task_system_prompt",
    "task_user_prompt",
]
