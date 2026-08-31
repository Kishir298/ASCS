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

from .config import AgentConfig, MODIFY_TOOLS, READONLY_TOOLS
from .events import (
    EventSink,
    emit_activity,
    emit_command_completed,
    emit_command_started,
    emit_status,
    emit_task_failed,
    emit_task_verified,
    null_sink,
)
from .models import ToolResult, parse_model_reply, tool_result_message
from .ollama import OllamaClient, OllamaError
from .prompts import system_prompt
from .tasks import COMPLETED, FAILED, PENDING, READY, RUNNING, SKIPPED, Task, TaskGraph
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
class TaskActionLog:
    """A single action taken during task execution (Phase 4.6 log contract)."""

    action: str = ""  # tool name or "verify"
    target: str = ""  # file path or command
    output: str = ""  # truncated result/note
    ok: bool = False


@dataclass
class TaskOutcome:
    """Result of executing (and verifying) one task."""

    task_id: str = ""
    ok: bool = False
    summary: str = ""
    iterations: int = 0
    verification: VerificationResult | None = None
    reason: str = ""
    action_log: list[TaskActionLog] = field(default_factory=list)


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

    @property
    def cancelled_count(self) -> int:
        return self.progress.get("cancelled", 0) + self.progress.get("skipped", 0)

    def files_changed(self) -> list[str]:
        """Files actually modified across all task action logs (deterministic)."""
        modifying = {
            "write_file", "edit_file", "apply_patch", "apply_diff",
            "delete_file", "move_file", "copy_file",
        }
        seen: dict[str, str] = {}
        for outcome in self.outcomes:
            if not outcome.ok:
                continue
            for entry in outcome.action_log:
                if not entry.ok or entry.action not in modifying:
                    continue
                target = entry.target.strip()
                if target:
                    seen[target] = target
        return sorted(seen)

    def total_retries(self) -> int:
        """Number of verification retry attempts actually performed."""
        return sum(
            1
            for outcome in self.outcomes
            for entry in outcome.action_log
            if entry.action == "verify" and not entry.ok
        ) if self.outcomes else 0

    def report(self) -> str:
        """A truthful, human-readable end-of-run report."""
        lines: list[str] = []
        lines.append(f"Objective: {self.objective}")
        lines.append(f"Status: {self.status}")
        lines.append(f"Tasks planned: {self.progress.get('total', 0)}")
        lines.append(f"Tasks completed: {self.progress.get('completed', 0)}")
        lines.append(f"Tasks failed: {self.progress.get('failed', 0)}")
        lines.append(f"Tasks cancelled/skipped: {self.cancelled_count}")
        changed = self.files_changed()
        if changed:
            lines.append("Files changed:")
            for path in changed:
                lines.append(f"  - {path}")
        else:
            lines.append("Files changed: (none recorded)")
        verified = sum(1 for o in self.outcomes if o.ok and o.verification)
        lines.append(f"Verification performed: {verified} task(s)")
        lines.append(f"Verification retries: {self.total_retries()}")
        if self.failed_tasks:
            lines.append("Remaining issues:")
            for outcome in self.failed_tasks:
                lines.append(f"  - {outcome.task_id}: {outcome.reason or outcome.summary}")
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        lines.append(
            "Achieved: "
            + ("yes" if self.status == "completed" else "no (incomplete or failed)")
        )
        return "\n".join(lines)


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
        approver: Callable[[str], bool] | None = None,
        git_baseline: set[str] | None = None,
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
        self.approver = approver
        self.git_baseline = git_baseline or set()
        self.max_verify_retries = config.max_verify_retries
        self.iterations = 0
        self._last_verify_failure: str = ""

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
                # Refresh project context (index + manifest) so later tasks
                # retrieve freshly-modified files rather than stale records.
                self._refresh_context()
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
        """Run one task to a terminal outcome, verifying it on success.

        Verification failures are retried up to ``self.max_verify_retries``
        times: the failure detail is fed back to the model so it can correct
        the implementation before the task is declared FAILED.
        """
        graph.mark(task.id, RUNNING)
        self.log(f"[task {task.id}] running: {task.title}")
        emit_status(self.event_sink, "EXECUTING", f"Task {task.id}: {task.title}")

        action_log: list[TaskActionLog] = []
        outcome = TaskOutcome(task_id=task.id)
        self._last_verify_failure = ""

        # -- run task (model loop) ---------------------------------------
        outcome = self.run_task(self, task)
        self.iterations += outcome.iterations
        action_log.extend(getattr(outcome, "action_log", []))

        if not outcome.ok:
            reason = outcome.reason or outcome.summary
            graph.mark(
                task.id, FAILED,
                retry_count=task.retry_count + 1,
                failure_reason=reason,
            )
            emit_status(self.event_sink, "FAILED", f"Task {task.id} failed: {reason}")
            emit_task_failed(self.event_sink, task.id, reason)
            outcome.action_log = action_log
            return outcome

        # -- verify with retries ----------------------------------------
        for attempt in range(1, self.max_verify_retries + 2):
            verification = self.verify(self, task)
            outcome.verification = verification

            if verification.ok:
                graph.mark(
                    task.id, COMPLETED,
                    retry_count=task.retry_count,
                    result=outcome.summary,
                )
                emit_status(self.event_sink, "EXECUTING", f"Task {task.id} verified")
                emit_task_verified(
                    self.event_sink, task.id, ok=True,
                    summary=outcome.summary,
                )
                action_log.append(TaskActionLog(
                    action="verify", target="acceptance criteria",
                    output="passed", ok=True,
                ))
                outcome.action_log = action_log
                return outcome

            # verification failed
            detail = verification.detail
            action_log.append(TaskActionLog(
                action="verify", target="acceptance criteria",
                output=f"attempt {attempt}: {detail[:200]}", ok=False,
            ))

            if attempt <= self.max_verify_retries:
                self._last_verify_failure = (
                    f"Verification failed (attempt {attempt}/"
                    f"{self.max_verify_retries + 1}):\n{detail}\n\n"
                    "Fix the issue above and finish with "
                    '{"done": true, "summary": "..."} so the task is re-verified.'
                )
                self.log(
                    f"[task {task.id}] verify attempt {attempt} failed, "
                    "retrying model loop..."
                )
                retry = self.run_task(self, task)
                self.iterations += retry.iterations
                action_log.extend(getattr(retry, "action_log", []))
                if not retry.ok:
                    reason = retry.reason or retry.summary
                    graph.mark(
                        task.id, FAILED,
                        retry_count=task.retry_count + 1,
                        failure_reason=reason,
                    )
                    emit_status(self.event_sink, "FAILED", f"Task {task.id} failed: {reason}")
                    emit_task_failed(self.event_sink, task.id, reason)
                    outcome.ok = False
                    outcome.reason = reason
                    outcome.action_log = action_log
                    return outcome
                outcome = retry
                self._last_verify_failure = ""

        # all retries exhausted
        reason = f"verification failed after {self.max_verify_retries + 1} attempt(s):\n{detail}"
        outcome.ok = False
        outcome.reason = reason
        graph.mark(
            task.id, FAILED,
            retry_count=task.retry_count + 1,
            failure_reason=reason,
        )
        emit_status(self.event_sink, "FAILED", f"Task {task.id} failed: {reason}")
        emit_task_failed(self.event_sink, task.id, reason)
        emit_task_verified(
            self.event_sink, task.id, ok=False,
            summary=reason,
        )
        outcome.action_log = action_log
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

    def _refresh_context(self) -> None:
        """Re-scan the project index so retrieval reflects post-task changes.

        Called after every task so later tasks in the same run see updated
        file content/dependencies. Best-effort: failures are surfaced to the
        log but never crash the run.
        """
        if self.store is None:
            return
        try:
            self.store.index.update()
        except Exception as exc:  # noqa: BLE001 - context refresh must not crash
            self.log(f"[executor] could not refresh project context: {exc}")

    # -- mode gating -------------------------------------------------------

    def _requires_approval(self, tool: str) -> bool:
        """True when the tool needs operator approval (SAFE mode + modifying tool)."""
        return self.config.is_safe_mode and tool in MODIFY_TOOLS

    def _check_tool_allowed(self, tool: str) -> ToolResult | None:
        """Check whether the current mode permits this tool.

        Returns ``None`` when the tool is allowed (proceed), or a
        ``ToolResult(ok=False)`` when it must be blocked.
        """
        if self.config.is_plan_mode and tool in MODIFY_TOOLS:
            return ToolResult(
                tool,
                f"Blocked in PLAN mode: {tool!r} modifies the workspace. "
                "Switch to BUILD or AUTO mode to make changes.",
                ok=False,
            )
        if self._requires_approval(tool):
            if self.approver is None:
                return ToolResult(
                    tool,
                    f"SAFE mode: {tool!r} requires operator approval but no approver "
                    "is configured. Pass --safe with an interactive session or "
                    "provide an approver callback.",
                    ok=False,
                )
            display = f"{tool}"
            if not self.approver(display):
                return ToolResult(
                    tool,
                    f"SAFE mode: operator declined {tool!r}.",
                    ok=False,
                )
        return None  # allowed

    def _check_git_dirty(self, tool: str, arguments: dict[str, Any]) -> ToolResult | None:
        """Block writes to files that were dirty before ASCS started (Phase 4.4).

        Only applies to modifying file tools; ``run_command`` is not gated
        here (it is gated by SAFE mode approval instead).
        """
        if not self.git_baseline or tool not in (
            "write_file", "apply_patch", "delete_file", "move_file", "copy_file",
        ):
            return None
        # Extract the target path from the tool arguments.
        target_path = arguments.get("path") or arguments.get("destination") or ""
        if not target_path or not isinstance(target_path, str):
            return None
        # Normalize: just the relative path as git reports it.
        target_rel = target_path.replace("\\", "/").strip()
        if target_rel in self.git_baseline:
            return ToolResult(
                tool,
                f"Protected: '{target_rel}' has pre-existing uncommitted changes. "
                "ASCS will not overwrite existing user work. Proceed with a "
                "different file or complete the current changes manually.",
                ok=False,
            )
        return None

    @staticmethod
    def reset_stuck_tasks(graph: TaskGraph) -> list[str]:
        """Reset any task stuck in RUNNING back to READY so resume can retry it.

        An interrupted run may leave a task marked RUNNING; a resumed run
        must not treat it as completed or block behind it.
        """
        reset: list[str] = []
        for task in list(graph.tasks.values()):
            if task.status == RUNNING:
                graph.mark(task.id, READY)
                reset.append(task.id)
        return reset

    # -- default single-task model loop ------------------------------------

    def _default_run_task(self, executor, task: Task) -> TaskOutcome:
        """A task-scoped agent loop; verified separately by ``_verify_task``.

        Keeps the same tool contract as the main loop but scoped to one task,
        bounded by the config's iteration budget. Tolerates malformed replies
        up to a limit, and never trusts a model "done" without verification.
        When the executor carries a ``_last_verify_failure`` (set on retry),
        the failure detail is prepended to the user message so the model can
        correct the implementation.
        """
        system = task_system_prompt(task, _project_block(executor.store), self.config)

        # Phase 4.7 retry: feed verification failure back to the model.
        failure_context = getattr(executor, "_last_verify_failure", "")

        user_msg = task_user_prompt(task)
        if failure_context:
            user_msg = (
                f"YOUR PREVIOUS ATTEMPT FAILED VERIFICATION — FIX THE ISSUE:\n"
                f"{failure_context}\n\n"
                f"Original task:\n{user_msg}"
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        outcome = TaskOutcome(task_id=task.id)
        action_log: list[TaskActionLog] = []

        for iteration in range(1, self.config.max_iterations + 1):
            if self.should_stop():
                outcome.ok = False
                outcome.reason = "cancelled by operator"
                outcome.action_log = action_log
                return outcome
            try:
                reply_text = _resilient_chat(
                    self.client,
                    _trim_for_request(messages, self.config.context_budget_chars),
                    format="json",
                    should_stop=self.should_stop,
                )
            except OllamaError as exc:
                outcome.ok = False
                outcome.reason = f"model error: {exc}"
                outcome.action_log = action_log
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
                outcome.action_log = action_log
                return outcome

            result = self._run_tool(reply.tool, reply.arguments, task, iteration)
            outcome.iterations = iteration
            # Record the action for the per-task action log (files/commands).
            _target = _action_target(reply.tool, reply.arguments)
            action_log.append(TaskActionLog(
                action=reply.tool,
                target=_target,
                output=result.note or ("" if result.ok else result.output[:200]),
                ok=result.ok,
            ))
            messages.append({"role": "assistant", "content": reply_text})
            messages.append(tool_result_message(result))

        outcome.ok = False
        outcome.reason = (
            f"Task exceeded {self.config.max_iterations} iterations without finishing"
        )
        outcome.action_log = action_log
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
        # Mode gate: PLAN = read-only; SAFE = approval before modify.
        blocked = self._check_tool_allowed(tool)
        if blocked is not None:
            self.log(f"[task {task.id}:{iteration:02d}] {tool}: BLOCKED ({blocked.output[:100]})")
            return blocked
        # Git-dirty guard: block writes to pre-existing dirty files.
        git_block = self._check_git_dirty(tool, arguments)
        if git_block is not None:
            self.log(f"[task {task.id}:{iteration:02d}] {tool}: GIT-BLOCKED ({git_block.output[:100]})")
            return git_block
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

        Phase 4.5/4.6: respects mode gating — PLAN mode blocks ``run_command``
        and modification commands; SAFE mode requires operator approval for
        modifying commands.  Phase 4.7: implementing tasks with no actionable
        verification steps are treated as *not fully verified*.
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

        # Phase 4.7 no-steps policy: implementing tasks must have actionable
        # verification or be treated as not fully verified.
        if not steps:
            _is_implementing = task.kind in ("implement", "")
            if _is_implementing:
                result.steps.append({
                    "step": "(no verification steps declared)",
                    "status": "failed",
                    "ok": False,
                    "output": "Implementing task has no verification steps; "
                              "add verification commands or checks to confirm "
                              "the task is correct.",
                })
                result.ok = False
                return result
            # Non-implementing tasks (inspect/plan/review) pass with no steps.
            result.ok = True
            return result

        all_ok = True
        for kind, payload in steps:
            if kind == "check":
                result.steps.append(
                    {"step": payload, "status": "noted", "ok": True, "output": ""}
                )
                continue
            # run_command in verify is gated like any other tool call.
            blocked = self._check_tool_allowed("run_command")
            if blocked is not None:
                result.steps.append({
                    "step": payload,
                    "status": "blocked",
                    "ok": False,
                    "output": blocked.output,
                })
                all_ok = False
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


def _action_target(tool: str, arguments: dict[str, Any]) -> str:
    """Best-effort target path/command for an action-log entry."""
    if tool == "run_command":
        return str(arguments.get("command", ""))
    for key in ("path", "destination", "files"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return "<no target>"


def _resilient_chat(client, messages: list[dict[str, str]], **kwargs) -> str:
    """Issue a chat request, using the resilient (retrying) path when available."""
    if hasattr(client, "chat_resilient"):
        return client.chat_resilient(messages, **kwargs)
    chat = getattr(client, "chat")
    kwargs.pop("should_stop", None)
    return chat(messages, **kwargs)


__all__ = [
    "TaskActionLog",
    "TaskExecutor",
    "TaskExecution",
    "TaskOutcome",
    "VerificationResult",
    "task_system_prompt",
    "task_user_prompt",
]
